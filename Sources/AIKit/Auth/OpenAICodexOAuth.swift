import Foundation

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// OpenAI's native-app authorization configuration for ChatGPT Codex.
public struct OpenAICodexOAuthConfiguration: Sendable, Hashable {
    public static let clientId = "app_EMoamEEZ73f0CkXaXp7hrann"
    public static let defaultAuthBaseURL = URL(string: "https://auth.openai.com")!

    public var clientId: String
    public var authBaseURL: URL
    public var browserRedirectURI: String
    public var originator: String

    public init(
        clientId: String = Self.clientId,
        authBaseURL: URL = Self.defaultAuthBaseURL,
        browserRedirectURI: String = "http://localhost:1455/auth/callback",
        originator: String = "aikit-swift"
    ) {
        self.clientId = clientId
        self.authBaseURL = authBaseURL
        self.browserRedirectURI = browserRedirectURI
        self.originator = originator
    }

    public var oauth: OAuthConfiguration {
        OAuthConfiguration(
            clientId: clientId,
            authorizationEndpoint: authBaseURL.appending(path: "oauth/authorize"),
            tokenEndpoint: authBaseURL.appending(path: "oauth/token"),
            redirectURI: browserRedirectURI,
            scopes: ["openid", "profile", "email", "offline_access"],
            additionalParameters: [
                "id_token_add_organizations": "true",
                "codex_cli_simplified_flow": "true",
                "originator": originator,
            ]
        )
    }

    public var deviceUserCodeURL: URL {
        authBaseURL.appending(path: "api/accounts/deviceauth/usercode")
    }

    public var deviceTokenURL: URL {
        authBaseURL.appending(path: "api/accounts/deviceauth/token")
    }

    public var deviceVerificationURL: URL {
        authBaseURL.appending(path: "codex/device")
    }

    public var deviceRedirectURI: String {
        authBaseURL.appending(path: "deviceauth/callback").absoluteString
    }
}

public struct OpenAICodexDeviceCode: Sendable, Hashable {
    public var deviceAuthId: String
    public var userCode: String
    public var verificationURL: URL
    public var interval: TimeInterval
    public var expiresIn: TimeInterval

    public init(
        deviceAuthId: String,
        userCode: String,
        verificationURL: URL,
        interval: TimeInterval,
        expiresIn: TimeInterval = 15 * 60
    ) {
        self.deviceAuthId = deviceAuthId
        self.userCode = userCode
        self.verificationURL = verificationURL
        self.interval = interval
        self.expiresIn = expiresIn
    }
}

public enum OpenAICodexOAuthError: Error, Sendable, LocalizedError {
    case unsuccessfulResponse(operation: String, status: Int, body: String)
    case malformedResponse(String)
    case missingAccountId
    case deviceFlowTimedOut
    case invalidBrowserRedirectURI
    case mismatchedRedirectListener

    public var errorDescription: String? {
        switch self {
        case .unsuccessfulResponse(let operation, let status, let body):
            "OpenAI Codex \(operation) failed (HTTP \(status)): \(body)"
        case .malformedResponse(let operation):
            "OpenAI Codex \(operation) returned a malformed response."
        case .missingAccountId:
            "Failed to extract the ChatGPT account ID from the OpenAI Codex access token."
        case .deviceFlowTimedOut:
            "OpenAI Codex device authorization timed out."
        case .invalidBrowserRedirectURI:
            "OpenAI Codex browser authorization requires a fixed HTTP loopback redirect URI."
        case .mismatchedRedirectListener:
            "The loopback listener does not match the registered OpenAI Codex redirect port and path."
        }
    }
}

/// Performs OpenAI Codex PKCE, device-code, exchange, and refresh operations.
///
/// `URLSession`, time, and sleeping are injectable so tests never need a real
/// login, browser, network connection, or wall-clock delay.
public struct OpenAICodexOAuthClient: Sendable {
    public typealias Sleep = @Sendable (TimeInterval) async throws -> Void

    public var configuration: OpenAICodexOAuthConfiguration
    public var session: URLSession
    private let now: @Sendable () -> Date
    private let sleep: Sleep
    private let credentialStore: any OAuthCredentialStore
    private let credentialKey: String
    private let deviceAuthorizationTimeout: TimeInterval

    public init(
        configuration: OpenAICodexOAuthConfiguration = .init(),
        session: URLSession = .shared,
        credentialStore: any OAuthCredentialStore = KeychainOAuthCredentialStore(),
        credentialKey: String = "openai-codex",
        deviceAuthorizationTimeout: TimeInterval = 15 * 60,
        now: @escaping @Sendable () -> Date = Date.init,
        sleep: @escaping Sleep = { seconds in
            try await Task.sleep(for: .seconds(seconds))
        }
    ) {
        self.configuration = configuration
        self.session = session
        self.credentialStore = credentialStore
        self.credentialKey = credentialKey
        self.deviceAuthorizationTimeout = deviceAuthorizationTimeout
        self.now = now
        self.sleep = sleep
    }

    /// Builds the exact OpenAI browser-login URL, including PKCE and Codex
    /// native-client parameters.
    public func authorizationURL(pkce: PKCE, state: String) -> URL {
        OAuthFlow.authorizationURL(configuration.oauth, pkce: pkce, state: state)
    }

    /// Runs browser login using the package's one-shot loopback listener.
    public func authorizePKCE(
        timeout: TimeInterval = 300,
        open: @Sendable (URL) -> Void
    ) async throws -> OAuthCredential {
        let redirect = try registeredLoopbackRedirect()
        let listener = try LoopbackRedirectListener(port: redirect.port, path: redirect.path)
        return try await authorizePKCE(listener: listener, timeout: timeout, open: open)
    }

    /// Runs browser login with an injected listener after proving that it
    /// matches OpenAI's registered fixed redirect.
    public func authorizePKCE(
        listener: LoopbackRedirectListener,
        timeout: TimeInterval = 300,
        open: @Sendable (URL) -> Void
    ) async throws -> OAuthCredential {
        let redirect = try registeredLoopbackRedirect()
        guard await listener.matches(port: redirect.port, path: redirect.path) else {
            await listener.stop()
            throw OpenAICodexOAuthError.mismatchedRedirectListener
        }
        let (code, pkce) = try await OAuthFlow.authorize(
            configuration.oauth,
            listener: listener,
            timeout: timeout,
            // OpenAI registers `localhost:1455`; the socket remains bound only
            // to loopback, but replacing this with 127.0.0.1 changes the OAuth
            // redirect value and causes the exchange to be rejected.
            registeredRedirectURI: configuration.browserRedirectURI,
            open: open
        )
        let credential = try await exchange(
            authorizationCode: code,
            pkce: pkce,
            redirectURI: configuration.browserRedirectURI
        )
        try await persistAuthorizedCredential(credential)
        return credential
    }

    public func exchange(
        authorizationCode: String,
        pkce: PKCE,
        redirectURI: String? = nil
    ) async throws -> OAuthCredential {
        guard Self.isPresent(configuration.clientId),
              Self.isPresent(authorizationCode),
              Self.isPresent(pkce.verifier)
        else {
            throw OpenAICodexOAuthError.malformedResponse("token exchange")
        }
        var oauth = configuration.oauth
        if let redirectURI { oauth.redirectURI = redirectURI }
        return try await sendTokenRequest(
            OAuthFlow.tokenExchange(oauth, code: authorizationCode, pkce: pkce),
            operation: "token exchange"
        )
    }

    public func refresh(_ credential: OAuthCredential) async throws -> OAuthCredential {
        guard Self.isPresent(configuration.clientId) else {
            throw OpenAICodexOAuthError.malformedResponse("token refresh")
        }
        guard let refreshToken = credential.refreshToken,
              Self.isPresent(refreshToken)
        else {
            throw OAuthCredentialManagerError.missingRefreshToken
        }
        let refreshed = try await sendTokenRequest(
            OAuthFlow.refresh(configuration.oauth, refreshToken: refreshToken),
            operation: "token refresh",
            fallbackCredential: credential
        )
        return refreshed
    }

    public func startDeviceAuthorization() async throws -> OpenAICodexDeviceCode {
        guard Self.isPresent(configuration.clientId) else {
            throw OpenAICodexOAuthError.malformedResponse("device-code request")
        }
        var request = URLRequest(url: configuration.deviceUserCodeURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.setValue("application/json", forHTTPHeaderField: "accept")
        request.httpBody = Data((try JSONValue.object([
            "client_id": .string(configuration.clientId)
        ]).encodedString()).utf8)

        let json = try await sendJSON(request, operation: "device-code request")
        guard let deviceAuthId = json["device_auth_id"]?.stringValue,
              Self.isPresent(deviceAuthId),
              let userCode = json["user_code"]?.stringValue,
              Self.isPresent(userCode),
              let interval = Self.number(json["interval"]), interval.isFinite, interval >= 0
        else {
            throw OpenAICodexOAuthError.malformedResponse("device-code request")
        }
        return OpenAICodexDeviceCode(
            deviceAuthId: deviceAuthId,
            userCode: userCode,
            verificationURL: configuration.deviceVerificationURL,
            interval: interval,
            expiresIn: deviceAuthorizationTimeout
        )
    }

    /// Completes the headless device flow and exchanges its authorization code.
    /// The first poll is immediate, matching OpenAI's Codex client behavior.
    public func authorizeDeviceCode(
        onCode: @escaping @Sendable (OpenAICodexDeviceCode) -> Void = { _ in }
    ) async throws -> OAuthCredential {
        let device = try await startDeviceAuthorization()
        onCode(device)

        let deadline = now().addingTimeInterval(device.expiresIn)
        var pollInterval = max(1, device.interval)
        while now() < deadline {
            try Task.checkCancellation()
            let result = try await pollDeviceAuthorization(device)
            switch result {
            case .pending:
                break
            case .slowDown:
                pollInterval += 5
            case .complete(let code, let verifier):
                let credential = try await exchange(
                    authorizationCode: code,
                    pkce: PKCE(verifier: verifier),
                    redirectURI: configuration.deviceRedirectURI
                )
                try await persistAuthorizedCredential(credential)
                return credential
            }

            let remaining = deadline.timeIntervalSince(now())
            guard remaining > 0 else { break }
            try await sleep(min(pollInterval, remaining))
        }
        throw OpenAICodexOAuthError.deviceFlowTimedOut
    }

    public func credentialManager(
        initialCredential: OAuthCredential? = nil,
        store: (any OAuthCredentialStore)? = nil,
        key: String? = nil,
        refreshLeeway: TimeInterval = 300
    ) -> OAuthCredentialManager {
        let client = self
        return OAuthCredentialManager(
            key: key ?? credentialKey,
            initialCredential: initialCredential,
            store: store ?? credentialStore,
            refreshLeeway: refreshLeeway,
            now: now
        ) { credential in
            try await client.refresh(credential)
        }
    }

    /// Extracts the account identity used by `chatgpt-account-id` from the
    /// access-token JWT. JWT signatures are verified by the TLS-protected OAuth
    /// issuer and backend; this local decode only reads the routing claim.
    public static func accountId(from accessToken: String) -> String? {
        guard let json = jwtPayload(from: accessToken),
              let accountId = json["https://api.openai.com/auth"]?["chatgpt_account_id"]?.stringValue,
              !accountId.isEmpty
        else { return nil }
        return accountId
    }

    private enum DevicePollResult {
        case pending
        case slowDown
        case complete(code: String, verifier: String)
    }

    private func pollDeviceAuthorization(
        _ device: OpenAICodexDeviceCode
    ) async throws -> DevicePollResult {
        guard Self.isPresent(device.deviceAuthId), Self.isPresent(device.userCode) else {
            throw OpenAICodexOAuthError.malformedResponse("device authorization poll")
        }
        var request = URLRequest(url: configuration.deviceTokenURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.setValue("application/json", forHTTPHeaderField: "accept")
        request.httpBody = Data((try JSONValue.object([
            "device_auth_id": .string(device.deviceAuthId),
            "user_code": .string(device.userCode),
        ]).encodedString()).utf8)

        let (data, response) = try await session.data(for: request)
        try Task.checkCancellation()
        guard let http = response as? HTTPURLResponse else {
            throw OpenAICodexOAuthError.malformedResponse("device authorization poll")
        }
        let json = try? JSONValue.decode(from: data)
        if (200..<300).contains(http.statusCode) {
            guard let code = json?["authorization_code"]?.stringValue,
                  Self.isPresent(code),
                  let verifier = json?["code_verifier"]?.stringValue,
                  Self.isPresent(verifier)
            else {
                throw OpenAICodexOAuthError.malformedResponse("device authorization poll")
            }
            return .complete(code: code, verifier: verifier)
        }

        let error = json?["error"]
        let code = error?.stringValue ?? error?["code"]?.stringValue
        if http.statusCode == 403 || http.statusCode == 404
            || code == "deviceauth_authorization_pending" {
            return .pending
        }
        if code == "slow_down" { return .slowDown }
        throw OpenAICodexOAuthError.unsuccessfulResponse(
            operation: "device authorization poll",
            status: http.statusCode,
            body: String(decoding: data, as: UTF8.self)
        )
    }

    private func sendTokenRequest(
        _ request: URLRequest,
        operation: String,
        fallbackCredential: OAuthCredential? = nil
    ) async throws -> OAuthCredential {
        let json = try await sendJSON(request, operation: operation)
        let currentTime = now()
        guard var credential = OAuthCredential(tokenResponse: json, now: currentTime) else {
            throw OpenAICodexOAuthError.malformedResponse(operation)
        }
        if let returnedRefreshToken = json["refresh_token"]?.stringValue {
            guard Self.isPresent(returnedRefreshToken) else {
                throw OpenAICodexOAuthError.malformedResponse(operation)
            }
            credential.refreshToken = returnedRefreshToken
        } else if let fallbackRefreshToken = fallbackCredential?.refreshToken,
                  Self.isPresent(fallbackRefreshToken) {
            credential.refreshToken = fallbackRefreshToken
        }
        guard let refreshToken = credential.refreshToken, Self.isPresent(refreshToken) else {
            throw OpenAICodexOAuthError.malformedResponse(operation)
        }
        if let lifetimeValue = json["expires_in"] {
            guard let lifetime = lifetimeValue.doubleValue,
                  lifetime.isFinite,
                  lifetime > 0
            else { throw OpenAICodexOAuthError.malformedResponse(operation) }
            credential.expiresAt = currentTime.addingTimeInterval(lifetime)
        } else {
            guard let tokenExpiry = Self.expiration(from: credential.accessToken),
                  tokenExpiry > currentTime
            else { throw OpenAICodexOAuthError.malformedResponse(operation) }
            credential.expiresAt = tokenExpiry
        }
        let accountId = Self.accountId(from: credential.accessToken)
            ?? json["id_token"]?.stringValue.flatMap(Self.accountId(from:))
            ?? fallbackCredential.flatMap { Self.accountId(from: $0.accessToken) }
            ?? fallbackCredential?.accountId
        guard let accountId, !accountId.isEmpty else {
            throw OpenAICodexOAuthError.missingAccountId
        }
        credential.accountId = accountId
        return credential
    }

    private func persistAuthorizedCredential(_ credential: OAuthCredential) async throws {
        try Task.checkCancellation()
        try await credentialStore.saveCredential(credential, for: credentialKey)
    }

    private func registeredLoopbackRedirect() throws -> (port: UInt16, path: String) {
        guard let components = URLComponents(string: configuration.browserRedirectURI),
              components.scheme?.lowercased() == "http",
              ["localhost", "127.0.0.1"].contains(components.host?.lowercased() ?? ""),
              let port = components.port,
              let fixedPort = UInt16(exactly: port),
              !components.path.isEmpty
        else { throw OpenAICodexOAuthError.invalidBrowserRedirectURI }
        return (fixedPort, components.path)
    }

    private func sendJSON(_ request: URLRequest, operation: String) async throws -> JSONValue {
        let (data, response) = try await session.data(for: request)
        try Task.checkCancellation()
        guard let http = response as? HTTPURLResponse else {
            throw OpenAICodexOAuthError.malformedResponse(operation)
        }
        guard (200..<300).contains(http.statusCode) else {
            throw OpenAICodexOAuthError.unsuccessfulResponse(
                operation: operation,
                status: http.statusCode,
                body: String(decoding: data, as: UTF8.self)
            )
        }
        guard let json = try? JSONValue.decode(from: data) else {
            throw OpenAICodexOAuthError.malformedResponse(operation)
        }
        return json
    }

    private static func number(_ value: JSONValue?) -> Double? {
        value?.doubleValue ?? value?.stringValue.flatMap(Double.init)
    }

    private static func isPresent(_ value: String) -> Bool {
        !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private static func expiration(from token: String) -> Date? {
        guard let payload = jwtPayload(from: token),
              let seconds = payload["exp"]?.doubleValue,
              seconds.isFinite
        else { return nil }
        return Date(timeIntervalSince1970: seconds)
    }

    private static func jwtPayload(from token: String) -> JSONValue? {
        let parts = token.split(separator: ".", omittingEmptySubsequences: false)
        guard parts.count == 3 else { return nil }
        var encoded = String(parts[1])
            .replacingOccurrences(of: "-", with: "+")
            .replacingOccurrences(of: "_", with: "/")
        let remainder = encoded.count % 4
        if remainder != 0 { encoded += String(repeating: "=", count: 4 - remainder) }
        guard let data = Data(base64Encoded: encoded) else { return nil }
        return try? JSONValue.decode(from: data)
    }
}
