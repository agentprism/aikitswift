import Foundation
import Network
import Testing

@testable import AIKit

/// The loopback listener is the one piece of this library that is genuinely
/// tested end to end: it binds a real socket and a real HTTP request is made
/// against it. No provider, no key, no network beyond localhost.
@Suite("Loopback redirect listener", .serialized)
struct LoopbackListenerTests {

    @Test("parses the target out of a request line")
    func parsesRequestTarget() {
        #expect(
            LoopbackRedirectListener.requestTarget("GET /callback?code=a&state=b HTTP/1.1\r\nHost: x\r\n\r\n")
                == "/callback?code=a&state=b"
        )
        // Bare LF, as some clients send.
        #expect(LoopbackRedirectListener.requestTarget("GET /cb HTTP/1.1\nHost: x") == "/cb")

        // Only GET is a redirect; anything else is something else entirely.
        #expect(LoopbackRedirectListener.requestTarget("POST /callback HTTP/1.1\r\n") == nil)
        #expect(LoopbackRedirectListener.requestTarget("garbage") == nil)
    }

    @Test("captures a real redirect over a real socket")
    func capturesRedirect() async throws {
        let listener = try LoopbackRedirectListener()
        let waiting = Task { try await listener.waitForRedirect(timeout: 10) }

        let port = try await listener.waitUntilBound()
        #expect(port != 0)

        // A real HTTP GET, exactly as a browser would send after the provider
        // redirects.
        let url = URL(string: "http://127.0.0.1:\(port)/callback?code=abc123&state=xyz")!
        let (data, response) = try await URLSession.shared.data(from: url)

        #expect((response as? HTTPURLResponse)?.statusCode == 200)
        // The browser must be shown something, or the user is left on a blank
        // error page wondering whether it worked.
        #expect(String(decoding: data, as: UTF8.self).contains("close this tab"))

        let captured = try await waiting.value
        let items = URLComponents(url: captured, resolvingAgainstBaseURL: false)?.queryItems ?? []
        #expect(items.first { $0.name == "code" }?.value == "abc123")
        #expect(items.first { $0.name == "state" }?.value == "xyz")
    }

    @Test("accepts a request line fragmented across TCP receives")
    func acceptsFragmentedRequestLine() async throws {
        let listener = try LoopbackRedirectListener()
        let waiting = Task { try await listener.waitForRedirect(timeout: 10) }
        let port = try await listener.waitUntilBound()
        let networkPort = try #require(NWEndpoint.Port(rawValue: port))
        let connection = NWConnection(host: .ipv4(.loopback), port: networkPort, using: .tcp)
        connection.start(queue: .global())
        defer { connection.cancel() }

        try await send(Data("GET /call".utf8), over: connection)
        // Force the listener's first receive to observe an incomplete request
        // line rather than relying on the TCP stack to coalesce both writes.
        try await Task.sleep(for: .milliseconds(100))
        try await send(Data(
            "back?code=fragmented&state=ok HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n".utf8
        ), over: connection)

        let captured = try await waiting.value
        let items = URLComponents(url: captured, resolvingAgainstBaseURL: false)?.queryItems ?? []
        #expect(items.first { $0.name == "code" }?.value == "fragmented")
    }

    @Test("rejects a wrong callback path before sending the success page")
    func rejectsWrongPathBeforeSuccessResponse() async throws {
        let listener = try LoopbackRedirectListener()
        let waiting = Task {
            try await listener.waitForRedirect(timeout: 10, expectedState: "expected")
        }
        let port = try await listener.waitUntilBound()
        let url = URL(string: "http://127.0.0.1:\(port)/wrong?code=x&state=expected")!

        let (data, response) = try await URLSession.shared.data(from: url)

        #expect((response as? HTTPURLResponse)?.statusCode == 404)
        #expect(String(decoding: data, as: UTF8.self).contains("Sign-in could not be completed"))
        await #expect(throws: LoopbackRedirectListener.Failure.self) {
            _ = try await waiting.value
        }
    }

    @Test("rejects a wrong OAuth state before sending the success page")
    func rejectsWrongStateBeforeSuccessResponse() async throws {
        let listener = try LoopbackRedirectListener()
        let waiting = Task {
            try await listener.waitForRedirect(timeout: 10, expectedState: "expected")
        }
        let port = try await listener.waitUntilBound()
        let url = URL(string: "http://127.0.0.1:\(port)/callback?code=x&state=wrong")!

        let (data, response) = try await URLSession.shared.data(from: url)

        #expect((response as? HTTPURLResponse)?.statusCode == 400)
        #expect(String(decoding: data, as: UTF8.self).contains("Sign-in could not be completed"))
        await #expect(throws: LoopbackRedirectListener.Failure.self) {
            _ = try await waiting.value
        }
    }

    @Test("binds only the loopback interface")
    func bindsLoopbackOnly() async throws {
        // An authorization code is a bearer credential for the seconds it
        // lives. A listener on a routable interface would accept it from
        // anyone on the network.
        let listener = try LoopbackRedirectListener()
        let waiting = Task { try await listener.waitForRedirect(timeout: 5) }
        let port = try await listener.waitUntilBound()

        let uri = try #require(await listener.redirectURI())
        #expect(uri == "http://127.0.0.1:\(port)/callback")

        await listener.stop()
        _ = try? await waiting.value
    }

    @Test("stopping resolves the wait rather than hanging")
    func stopUnblocks() async throws {
        let listener = try LoopbackRedirectListener()
        let waiting = Task { try await listener.waitForRedirect(timeout: 30) }
        try await listener.waitUntilBound()

        await listener.stop()

        await #expect(throws: LoopbackRedirectListener.Failure.self) {
            _ = try await waiting.value
        }
    }

    @Test("a user who wanders off does not leave a port bound forever")
    func timesOut() async throws {
        let listener = try LoopbackRedirectListener()

        await #expect(throws: LoopbackRedirectListener.Failure.self) {
            _ = try await listener.waitForRedirect(timeout: 0.2)
        }
    }

    @Test("the full authorization flow yields a code")
    func runsFullFlow() async throws {
        let listener = try LoopbackRedirectListener()
        let configuration = OAuthConfiguration(
            clientId: "client-1",
            authorizationEndpoint: URL(string: "https://example.com/authorize")!,
            tokenEndpoint: URL(string: "https://example.com/token")!,
            // Overwritten with the bound port — the ephemeral port is not known
            // until the listener is up.
            redirectURI: "http://127.0.0.1:0/callback",
            scopes: ["read"]
        )

        let (code, pkce) = try await OAuthFlow.authorize(
            configuration,
            listener: listener,
            timeout: 10
        ) { url in
            // Stands in for a browser: read the state the flow generated and
            // redirect back the way a provider would.
            let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
            let state = items.first { $0.name == "state" }?.value ?? ""
            let redirect = items.first { $0.name == "redirect_uri" }?.value ?? ""

            Task {
                let callback = URL(string: "\(redirect)?code=granted-code&state=\(state)")!
                _ = try? await URLSession.shared.data(from: callback)
            }
        }

        #expect(code == "granted-code")
        // The verifier is needed for the token exchange that follows.
        #expect(!pkce.verifier.isEmpty)
    }

    @Test("a redirect carrying the wrong state is rejected")
    func rejectsForeignRedirect() async throws {
        // Without the state check, a redirect from an unrelated flow would be
        // accepted as this one's.
        let listener = try LoopbackRedirectListener()
        let configuration = OAuthConfiguration(
            clientId: "client-1",
            authorizationEndpoint: URL(string: "https://example.com/authorize")!,
            tokenEndpoint: URL(string: "https://example.com/token")!,
            redirectURI: "http://127.0.0.1:0/callback"
        )

        await #expect(throws: LoopbackRedirectListener.Failure.self) {
            _ = try await OAuthFlow.authorize(configuration, listener: listener, timeout: 10) { url in
                let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
                let redirect = items.first { $0.name == "redirect_uri" }?.value ?? ""

                Task {
                    let callback = URL(string: "\(redirect)?code=stolen&state=not-our-state")!
                    _ = try? await URLSession.shared.data(from: callback)
                }
            }
        }
    }

    @Test("cancelling authorization stops its listener and releases the port")
    func authorizationCancellationCleansUpListener() async throws {
        let listener = try LoopbackRedirectListener()
        let configuration = OAuthConfiguration(
            clientId: "client-1",
            authorizationEndpoint: URL(string: "https://example.com/authorize")!,
            tokenEndpoint: URL(string: "https://example.com/token")!,
            redirectURI: "http://127.0.0.1:0/callback"
        )
        let authorization = Task {
            try await OAuthFlow.authorize(
                configuration,
                listener: listener,
                timeout: 30
            ) { _ in }
        }
        let port = try await listener.waitUntilBound()

        authorization.cancel()
        do {
            _ = try await authorization.value
            Issue.record("Expected cancellation")
        } catch {
            #expect(error is CancellationError)
        }

        let replacement = try LoopbackRedirectListener(port: port)
        let replacementWait = Task { try await replacement.waitForRedirect(timeout: 30) }
        #expect(try await replacement.waitUntilBound() == port)
        await replacement.stop()
        _ = try? await replacementWait.value
    }

    @Test("an early listener failure resolves authorization and releases the port")
    func earlyAuthorizationFailureCleansUpListener() async throws {
        let listener = try LoopbackRedirectListener()
        let initialWait = Task { try await listener.waitForRedirect(timeout: 30) }
        let port = try await listener.waitUntilBound()
        await listener.stop()
        _ = try? await initialWait.value

        let configuration = OAuthConfiguration(
            clientId: "client-1",
            authorizationEndpoint: URL(string: "https://example.com/authorize")!,
            tokenEndpoint: URL(string: "https://example.com/token")!,
            redirectURI: "http://127.0.0.1:0/callback"
        )
        await #expect(throws: LoopbackRedirectListener.Failure.self) {
            _ = try await OAuthFlow.authorize(
                configuration,
                listener: listener,
                timeout: 30
            ) { _ in }
        }

        let replacement = try LoopbackRedirectListener(port: port)
        let replacementWait = Task { try await replacement.waitForRedirect(timeout: 30) }
        #expect(try await replacement.waitUntilBound() == port)
        await replacement.stop()
        _ = try? await replacementWait.value
    }

    private func send(_ data: Data, over connection: NWConnection) async throws {
        try await withCheckedThrowingContinuation {
            (continuation: CheckedContinuation<Void, any Error>) in
            connection.send(content: data, completion: .contentProcessed { error in
                if let error {
                    continuation.resume(throwing: error)
                } else {
                    continuation.resume()
                }
            })
        }
    }
}
