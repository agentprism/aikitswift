import CryptoKit
import Foundation

/// Proof Key for Code Exchange (RFC 7636).
///
/// Required for any OAuth flow in a native app, because a desktop or mobile
/// binary cannot keep a client secret: anyone can extract it. PKCE replaces the
/// secret with a per-request one-time proof, so intercepting the redirect is
/// not enough to redeem the code.
public struct PKCE: Sendable, Hashable {
    /// The high-entropy secret, held until the token exchange.
    public let verifier: String
    /// The SHA-256 hash sent with the authorization request.
    public let challenge: String
    public let method = "S256"

    /// Generates a fresh pair.
    ///
    /// - Parameter length: verifier length, 43–128 per the RFC.
    public init(length: Int = 64) {
        let alphabet = Array("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
        let clamped = min(max(length, 43), 128)

        var verifier = ""
        verifier.reserveCapacity(clamped)
        for _ in 0..<clamped {
            // The system CSPRNG. A predictable verifier defeats the point.
            verifier.append(alphabet[Int.random(in: 0..<alphabet.count)])
        }

        self.verifier = verifier
        self.challenge = Self.challenge(for: verifier)
    }

    init(verifier: String) {
        self.verifier = verifier
        self.challenge = Self.challenge(for: verifier)
    }

    /// base64url(SHA-256(verifier)), unpadded — the encoding the RFC mandates.
    static func challenge(for verifier: String) -> String {
        let digest = SHA256.hash(data: Data(verifier.utf8))
        return Data(digest).base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
}

/// An OAuth endpoint configuration.
///
/// Client ids and endpoints are supplied by the caller rather than baked in:
/// they are the integrating application's identity, they change without notice,
/// and shipping someone else's in a library is both fragile and impolite.
public struct OAuthConfiguration: Sendable, Hashable {
    public var clientId: String
    public var authorizationEndpoint: URL
    public var tokenEndpoint: URL
    public var redirectURI: String
    public var scopes: [String]
    /// Extra query items for the authorization request.
    public var additionalParameters: [String: String]

    public init(
        clientId: String,
        authorizationEndpoint: URL,
        tokenEndpoint: URL,
        redirectURI: String,
        scopes: [String] = [],
        additionalParameters: [String: String] = [:]
    ) {
        self.clientId = clientId
        self.authorizationEndpoint = authorizationEndpoint
        self.tokenEndpoint = tokenEndpoint
        self.redirectURI = redirectURI
        self.scopes = scopes
        self.additionalParameters = additionalParameters
    }
}

/// A stored OAuth credential.
public struct OAuthCredential: Sendable, Hashable, Codable {
    public var accessToken: String
    public var refreshToken: String?
    public var expiresAt: Date?
    public var scopes: [String]
    /// Provider account identity carried alongside the token when the wire
    /// protocol needs more than bearer authorization. ChatGPT Codex is the
    /// current example: its backend requires `chatgpt-account-id` on every
    /// request, and refreshed access tokens may select a different account.
    public var accountId: String?

    public init(
        accessToken: String,
        refreshToken: String? = nil,
        expiresAt: Date? = nil,
        scopes: [String] = [],
        accountId: String? = nil
    ) {
        self.accessToken = accessToken
        self.refreshToken = refreshToken
        self.expiresAt = expiresAt
        self.scopes = scopes
        self.accountId = accountId
    }

    /// Whether the token should be refreshed before the next request.
    ///
    /// The leeway matters: a token that is valid *now* can expire while the
    /// request is in flight, and a long streaming response makes that window
    /// much wider than it looks.
    public func needsRefresh(now: Date = Date(), leeway: TimeInterval = 300) -> Bool {
        guard let expiresAt else { return false }
        return now.addingTimeInterval(leeway) >= expiresAt
    }

    /// Builds a credential from a token endpoint's JSON response.
    public init?(tokenResponse: JSONValue, now: Date = Date()) {
        guard let accessToken = tokenResponse["access_token"]?.stringValue,
              !accessToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { return nil }

        self.accessToken = accessToken
        self.refreshToken = tokenResponse["refresh_token"]?.stringValue
        // Servers report a lifetime, not a deadline.
        self.expiresAt = tokenResponse["expires_in"]?.doubleValue
            .map { now.addingTimeInterval($0) }
        self.scopes = tokenResponse["scope"]?.stringValue?
            .split(separator: " ").map(String.init) ?? []
        self.accountId = nil
    }
}

/// Builds the requests of an authorization-code-with-PKCE flow.
///
/// Deliberately does no I/O. Opening a browser, running a loopback listener and
/// storing secrets are all host concerns with platform-specific answers; this
/// type shapes the requests and interprets the responses, which is the part
/// that is easy to get subtly wrong and easy to test.
public enum OAuthFlow {

    /// The URL to send the user to.
    public static func authorizationURL(
        _ configuration: OAuthConfiguration,
        pkce: PKCE,
        state: String
    ) -> URL {
        var components = URLComponents(
            url: configuration.authorizationEndpoint,
            resolvingAgainstBaseURL: false
        )!

        var items = [
            URLQueryItem(name: "response_type", value: "code"),
            URLQueryItem(name: "client_id", value: configuration.clientId),
            URLQueryItem(name: "redirect_uri", value: configuration.redirectURI),
            URLQueryItem(name: "code_challenge", value: pkce.challenge),
            URLQueryItem(name: "code_challenge_method", value: pkce.method),
            // Echoed back on the redirect; a mismatch means the response
            // belongs to a different request and must be rejected.
            URLQueryItem(name: "state", value: state),
        ]

        if !configuration.scopes.isEmpty {
            items.append(URLQueryItem(name: "scope", value: configuration.scopes.joined(separator: " ")))
        }
        for (name, value) in configuration.additionalParameters.sorted(by: { $0.key < $1.key }) {
            items.append(URLQueryItem(name: name, value: value))
        }

        components.queryItems = items
        return components.url!
    }

    /// The request that redeems an authorization code for tokens.
    public static func tokenExchange(
        _ configuration: OAuthConfiguration,
        code: String,
        pkce: PKCE
    ) -> URLRequest {
        form(configuration.tokenEndpoint, fields: [
            "grant_type": "authorization_code",
            "code": code,
            "client_id": configuration.clientId,
            "redirect_uri": configuration.redirectURI,
            // Proves this client started the flow.
            "code_verifier": pkce.verifier,
        ])
    }

    /// The request that exchanges a refresh token for a new access token.
    public static func refresh(
        _ configuration: OAuthConfiguration,
        refreshToken: String
    ) -> URLRequest {
        form(configuration.tokenEndpoint, fields: [
            "grant_type": "refresh_token",
            "refresh_token": refreshToken,
            "client_id": configuration.clientId,
        ])
    }

    /// Extracts the code from a redirect, validating `state`.
    ///
    /// Returns `nil` when the state does not match. Skipping that check is the
    /// classic authorization-code interception bug.
    public static func code(fromRedirect url: URL, expectedState: String) -> String? {
        guard let items = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems,
              items.first(where: { $0.name == "state" })?.value == expectedState
        else { return nil }

        return items.first { $0.name == "code" }?.value
    }

    /// Token endpoints take form encoding, not JSON.
    private static func form(_ url: URL, fields: [String: String]) -> URLRequest {
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "content-type")
        request.setValue("application/json", forHTTPHeaderField: "accept")

        // Sorted so a request is reproducible and testable.
        var components = URLComponents()
        components.queryItems = fields.sorted { $0.key < $1.key }
            .map { URLQueryItem(name: $0.key, value: $0.value) }
        request.httpBody = Data((components.percentEncodedQuery ?? "").utf8)

        return request
    }
}
