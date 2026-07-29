import Foundation
import Network

/// A one-shot local HTTP listener that captures an OAuth redirect.
///
/// The browser half of an authorization-code flow needs somewhere for the
/// provider to redirect back to. For a native app that is a loopback listener:
/// the user authorizes in a real browser, the provider redirects to
/// `http://127.0.0.1:<port>/callback?code=…`, and this captures it.
///
/// Bound to `127.0.0.1` deliberately — not `0.0.0.0`. An authorization code is
/// a bearer credential for the seconds it lives, and a listener on a routable
/// interface would accept it from anyone on the network.
///
/// Single-use: it serves one request and stops. Anything else would leave a
/// port open long after the flow completed.
public actor LoopbackRedirectListener {

    public enum Failure: Error, CustomStringConvertible {
        case cannotBind(String)
        case timedOut
        case cancelled
        case malformedRequest

        public var description: String {
            switch self {
            case .cannotBind(let detail): "Could not bind a loopback port: \(detail)"
            case .timedOut: "No redirect arrived before the timeout."
            case .cancelled: "The listener was stopped before a redirect arrived."
            case .malformedRequest: "The redirect was not a well-formed HTTP request."
            }
        }
    }

    private let listener: NWListener
    private let path: String
    private var continuation: CheckedContinuation<URL, any Error>?
    private var finished = false

    /// The page shown in the browser once the redirect is captured.
    public var successHTML =
        "<!doctype html><meta charset=utf-8><title>Signed in</title>"
        + "<body style=\"font:16px system-ui;padding:3rem\">Signed in. You can close this tab."

    /// - Parameters:
    ///   - port: the port to bind, or `nil` for an ephemeral one. Providers
    ///     usually require the redirect URI to be registered in advance, which
    ///     means a fixed port.
    ///   - path: the redirect path, matched against the incoming request.
    public init(port: UInt16? = nil, path: String = "/callback") throws {
        self.path = path

        let parameters = NWParameters.tcp
        // Without this a listener lingering in TIME_WAIT blocks a retry on the
        // same fixed port, which is exactly when a user tries again.
        parameters.allowLocalEndpointReuse = true
        parameters.requiredLocalEndpoint = .hostPort(
            host: .ipv4(.loopback),
            port: port.flatMap { NWEndpoint.Port(rawValue: $0) } ?? .any
        )

        do {
            self.listener = try NWListener(using: parameters)
        } catch {
            throw Failure.cannotBind(String(describing: error))
        }
    }

    /// The port actually bound. Only meaningful once ``waitForRedirect(timeout:)``
    /// has started the listener.
    public var boundPort: UInt16? {
        listener.port?.rawValue
    }

    /// The redirect URI to send with the authorization request.
    public func redirectURI() -> String? {
        boundPort.map { "http://127.0.0.1:\($0)\(path)" }
    }

    /// Waits until the port is actually bound.
    ///
    /// Binding is asynchronous, and with an ephemeral port the number is not
    /// known until it completes — yet it is part of the redirect URI, so the
    /// authorization URL cannot be built before this returns.
    @discardableResult
    public func waitUntilBound(timeout: TimeInterval = 5) async throws -> UInt16 {
        let deadline = Date().addingTimeInterval(timeout)

        while Date() < deadline {
            if let port = listener.port?.rawValue, port != 0 { return port }
            try await Task.sleep(for: .milliseconds(10))
        }
        throw Failure.cannotBind("timed out waiting for the port to bind")
    }

    /// Starts listening and resumes when a redirect arrives.
    ///
    /// - Parameter timeout: how long to wait. A user who wanders off mid-login
    ///   should not leave a port bound indefinitely.
    public func waitForRedirect(timeout: TimeInterval = 300) async throws -> URL {
        // The timeout resumes the continuation rather than racing it in a task
        // group. A group would deadlock: `withCheckedThrowingContinuation` does
        // not observe cancellation, so cancelling the losing child leaves the
        // group waiting on a continuation nothing will ever resume.
        let timeoutTask = Task { [weak self] in
            do {
                try await Task.sleep(for: .seconds(timeout))
            } catch {
                return  // cancelled because the redirect already arrived
            }
            await self?.fail(.timedOut)
        }
        defer { timeoutTask.cancel() }

        let url = try await listen()
        stop()
        return url
    }

    /// Stops the listener. Safe to call more than once.
    public func stop() {
        listener.cancel()
        if let continuation {
            self.continuation = nil
            finished = true
            continuation.resume(throwing: Failure.cancelled)
        }
    }

    // MARK: - Internals

    private func listen() async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            self.continuation = continuation

            listener.newConnectionHandler = { [weak self] connection in
                connection.start(queue: .global())
                Task { await self?.receive(on: connection) }
            }

            listener.stateUpdateHandler = { [weak self] state in
                guard case .failed(let error) = state else { return }
                Task { await self?.fail(.cannotBind(String(describing: error))) }
            }

            listener.start(queue: .global())
        }
    }

    private func receive(on connection: NWConnection) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 8192) { [weak self] data, _, _, _ in
            guard let self else { return }

            Task {
                guard let data, let request = String(data: data, encoding: .utf8) else {
                    await self.fail(.malformedRequest)
                    connection.cancel()
                    return
                }

                let html = await self.successHTML
                // Reply before resolving, so the browser shows the page even
                // though the listener is about to stop.
                let response = """
                    HTTP/1.1 200 OK\r
                    Content-Type: text/html; charset=utf-8\r
                    Content-Length: \(html.utf8.count)\r
                    Connection: close\r
                    \r
                    \(html)
                    """
                connection.send(content: Data(response.utf8), completion: .contentProcessed { _ in
                    connection.cancel()
                })

                await self.handle(request: request)
            }
        }
    }

    private func handle(request: String) {
        guard let target = Self.requestTarget(request) else {
            fail(.malformedRequest)
            return
        }

        // The request line carries only a path and query; a scheme and host are
        // needed to make it a URL the flow can read query items from.
        guard let url = URL(string: "http://127.0.0.1\(target)") else {
            fail(.malformedRequest)
            return
        }

        succeed(url)
    }

    /// Extracts the target from an HTTP request line: `GET /callback?… HTTP/1.1`.
    static func requestTarget(_ request: String) -> String? {
        guard let line = request.split(separator: "\r\n", maxSplits: 1).first
            ?? request.split(separator: "\n", maxSplits: 1).first
        else { return nil }

        let fields = line.split(separator: " ")
        guard fields.count >= 2, fields[0] == "GET" else { return nil }
        return String(fields[1])
    }

    private func succeed(_ url: URL) {
        guard !finished, let continuation else { return }
        self.continuation = nil
        finished = true
        continuation.resume(returning: url)
    }

    private func fail(_ error: Failure) {
        guard !finished, let continuation else { return }
        self.continuation = nil
        finished = true
        continuation.resume(throwing: error)
    }
}

extension OAuthFlow {
    /// Runs the browser half of an authorization-code flow end to end.
    ///
    /// Opening a URL is platform-specific — `NSWorkspace` on macOS,
    /// `UIApplication` on iOS, `ASWebAuthenticationSession` where a private
    /// session is wanted — so the host supplies it rather than this library
    /// picking one and dragging in a UI framework.
    ///
    /// - Returns: the authorization code, ready for ``tokenExchange(_:code:pkce:)``.
    public static func authorize(
        _ configuration: OAuthConfiguration,
        listener: LoopbackRedirectListener,
        timeout: TimeInterval = 300,
        open: @Sendable (URL) -> Void
    ) async throws -> (code: String, pkce: PKCE) {
        let pkce = PKCE()
        // Unguessable, and checked on the way back: without it a redirect from
        // an unrelated flow would be accepted as this one's.
        let state = UUID().uuidString

        // The listener must be bound before the URL is built, because the
        // ephemeral port is part of the redirect URI.
        let listening = Task { try await listener.waitForRedirect(timeout: timeout) }
        try await listener.waitUntilBound()

        var resolved = configuration
        if let uri = await listener.redirectURI() {
            resolved.redirectURI = uri
        }

        open(authorizationURL(resolved, pkce: pkce, state: state))

        let redirect = try await listening.value
        guard let code = code(fromRedirect: redirect, expectedState: state) else {
            throw LoopbackRedirectListener.Failure.malformedRequest
        }

        return (code, pkce)
    }
}
