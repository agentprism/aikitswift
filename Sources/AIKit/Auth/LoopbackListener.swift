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

    public enum Failure: Error, CustomStringConvertible, Sendable {
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
    private let requestedPort: UInt16?
    private var continuation: CheckedContinuation<URL, any Error>?
    private var finished = false
    private var terminalFailure: Failure?
    private var expectedState: String?

    private static let maximumRequestLineBytes = 8192
    private static let failureHTML =
        "<!doctype html><meta charset=utf-8><title>Sign-in failed</title>"
        + "<body style=\"font:16px system-ui;padding:3rem\">"
        + "Sign-in could not be completed. Return to the app and try again."

    private enum RedirectResult: Sendable {
        case success(URL)
        case failure(Failure)
    }

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
        self.requestedPort = port

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

    /// The port actually bound. Only meaningful once
    /// ``waitForRedirect(timeout:expectedState:)``
    /// has started the listener.
    public var boundPort: UInt16? {
        listener.port?.rawValue
    }

    /// The redirect URI to send with the authorization request.
    public func redirectURI() -> String? {
        boundPort.map { "http://127.0.0.1:\($0)\(path)" }
    }

    /// Whether this listener was constructed for a registered fixed redirect.
    public func matches(port: UInt16, path: String) -> Bool {
        requestedPort == port && self.path == path
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
            if let terminalFailure { throw terminalFailure }
            if let port = listener.port?.rawValue, port != 0 { return port }
            try await Task.sleep(for: .milliseconds(10))
        }
        throw Failure.cannotBind("timed out waiting for the port to bind")
    }

    /// Starts listening and resumes when a redirect arrives.
    ///
    /// - Parameters:
    ///   - timeout: how long to wait. A user who wanders off mid-login should
    ///     not leave a port bound indefinitely.
    ///   - expectedState: when supplied, the redirect must carry exactly this
    ///     OAuth state value before the listener sends its success page.
    public func waitForRedirect(
        timeout: TimeInterval = 300,
        expectedState: String? = nil
    ) async throws -> URL {
        // The timeout resumes the continuation rather than racing it in a task
        // group. A group would deadlock: `withCheckedThrowingContinuation` does
        // not observe cancellation, so cancelling the losing child leaves the
        // group waiting on a continuation nothing will ever resume.
        try Task.checkCancellation()
        self.expectedState = expectedState
        return try await withTaskCancellationHandler {
            let timeoutTask = Task { [weak self] in
                do {
                    try await Task.sleep(for: .seconds(timeout))
                } catch {
                    return  // cancelled because the redirect already arrived
                }
                await self?.fail(.timedOut)
            }
            defer { timeoutTask.cancel() }

            do {
                let url = try await listen()
                stop()
                try Task.checkCancellation()
                return url
            } catch {
                stop()
                if Task.isCancelled { throw CancellationError() }
                throw error
            }
        } onCancel: {
            Task { await self.stop() }
        }
    }

    /// Stops the listener. Safe to call more than once.
    public func stop() {
        listener.cancel()
        terminalFailure = terminalFailure ?? .cancelled
        finished = true
        if let continuation {
            self.continuation = nil
            continuation.resume(throwing: Failure.cancelled)
        }
    }

    // MARK: - Internals

    private func listen() async throws -> URL {
        try await withCheckedThrowingContinuation { continuation in
            if let terminalFailure {
                continuation.resume(throwing: terminalFailure)
                return
            }
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

    private func receive(on connection: NWConnection, requestLine: Data = Data()) {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 2048) {
            [weak self] data, _, isComplete, error in
            guard let self else { return }
            let transportFailed = error != nil

            Task {
                await self.consume(
                    data,
                    on: connection,
                    requestLine: requestLine,
                    isComplete: isComplete,
                    transportFailed: transportFailed
                )
            }
        }
    }

    private func consume(
        _ data: Data?,
        on connection: NWConnection,
        requestLine: Data,
        isComplete: Bool,
        transportFailed: Bool
    ) {
        var buffered = requestLine
        if let data { buffered.append(data) }

        if let newline = buffered.firstIndex(of: UInt8(ascii: "\n")) {
            let line = buffered.prefix(through: newline)
            guard line.count <= Self.maximumRequestLineBytes,
                  let request = String(data: line, encoding: .utf8)
            else {
                respond(
                    status: "400 Bad Request",
                    html: Self.failureHTML,
                    result: .failure(.malformedRequest),
                    on: connection
                )
                return
            }
            handle(request: request, on: connection)
            return
        }

        guard buffered.count <= Self.maximumRequestLineBytes,
              !isComplete,
              !transportFailed
        else {
            respond(
                status: "400 Bad Request",
                html: Self.failureHTML,
                result: .failure(.malformedRequest),
                on: connection
            )
            return
        }

        // TCP is a byte stream: even the request line can arrive in multiple
        // receives. Keep reading until its LF delimiter rather than treating
        // the first packet as the complete HTTP request.
        receive(on: connection, requestLine: buffered)
    }

    private func handle(request: String, on connection: NWConnection) {
        guard let target = Self.requestTarget(request) else {
            respond(
                status: "400 Bad Request",
                html: Self.failureHTML,
                result: .failure(.malformedRequest),
                on: connection
            )
            return
        }

        // The request line carries only a path and query; a scheme and host are
        // needed to make it a URL the flow can read query items from.
        guard let url = URL(string: "http://127.0.0.1\(target)") else {
            respond(
                status: "400 Bad Request",
                html: Self.failureHTML,
                result: .failure(.malformedRequest),
                on: connection
            )
            return
        }
        guard url.path == path else {
            respond(
                status: "404 Not Found",
                html: Self.failureHTML,
                result: .failure(.malformedRequest),
                on: connection
            )
            return
        }
        if let expectedState {
            let states = URLComponents(url: url, resolvingAgainstBaseURL: false)?
                .queryItems?.filter { $0.name == "state" } ?? []
            guard states.count == 1, states[0].value == expectedState else {
                respond(
                    status: "400 Bad Request",
                    html: Self.failureHTML,
                    result: .failure(.malformedRequest),
                    on: connection
                )
                return
            }
        }

        respond(status: "200 OK", html: successHTML, result: .success(url), on: connection)
    }

    private func respond(
        status: String,
        html: String,
        result: RedirectResult,
        on connection: NWConnection
    ) {
        let response = """
            HTTP/1.1 \(status)\r
            Content-Type: text/html; charset=utf-8\r
            Content-Length: \(html.utf8.count)\r
            Cache-Control: no-store\r
            Connection: close\r
            \r
            \(html)
            """
        connection.send(content: Data(response.utf8), completion: .contentProcessed { [weak self] _ in
            connection.cancel()
            guard let self else { return }
            Task { await self.complete(result) }
        })
    }

    private func complete(_ result: RedirectResult) {
        switch result {
        case .success(let url): succeed(url)
        case .failure(let error): fail(error)
        }
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
        guard !finished else { return }
        terminalFailure = error
        finished = true
        guard let continuation else { return }
        self.continuation = nil
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
        registeredRedirectURI: String? = nil,
        open: @Sendable (URL) -> Void
    ) async throws -> (code: String, pkce: PKCE) {
        try Task.checkCancellation()
        let pkce = PKCE()
        // Unguessable, and checked on the way back: without it a redirect from
        // an unrelated flow would be accepted as this one's.
        let state = UUID().uuidString

        // The listener must be bound before the URL is built, because the
        // ephemeral port is part of the redirect URI.
        let listening = Task {
            try await listener.waitForRedirect(timeout: timeout, expectedState: state)
        }
        return try await withTaskCancellationHandler {
            do {
                try await listener.waitUntilBound()

                var resolved = configuration
                if let registeredRedirectURI {
                    resolved.redirectURI = registeredRedirectURI
                } else if let uri = await listener.redirectURI() {
                    resolved.redirectURI = uri
                }

                try Task.checkCancellation()
                open(authorizationURL(resolved, pkce: pkce, state: state))

                let redirect = try await listening.value
                try Task.checkCancellation()
                guard let code = code(fromRedirect: redirect, expectedState: state) else {
                    throw LoopbackRedirectListener.Failure.malformedRequest
                }

                return (code, pkce)
            } catch {
                listening.cancel()
                await listener.stop()
                _ = await listening.result
                if Task.isCancelled { throw CancellationError() }
                throw error
            }
        } onCancel: {
            listening.cancel()
            Task { await listener.stop() }
        }
    }
}
