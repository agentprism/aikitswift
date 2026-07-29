import Foundation
import Testing

@testable import Manifold

@Suite("OAuth")
struct OAuthTests {

    static let configuration = OAuthConfiguration(
        clientId: "client-123",
        authorizationEndpoint: URL(string: "https://example.com/authorize")!,
        tokenEndpoint: URL(string: "https://example.com/token")!,
        redirectURI: "http://localhost:8765/callback",
        scopes: ["read", "write"]
    )

    // MARK: - PKCE

    @Test("verifier length stays inside the RFC bounds")
    func clampsVerifierLength() {
        #expect(PKCE(length: 64).verifier.count == 64)
        // 43–128 per RFC 7636; out-of-range requests are clamped rather than
        // producing a verifier the server will reject.
        #expect(PKCE(length: 10).verifier.count == 43)
        #expect(PKCE(length: 500).verifier.count == 128)
    }

    @Test("verifiers use only unreserved characters")
    func usesUnreservedCharacters() {
        let allowed = CharacterSet(charactersIn: "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-._~")
        // Anything outside this set has to be percent-encoded, and servers
        // differ on whether they decode before comparing.
        #expect(PKCE().verifier.unicodeScalars.allSatisfy { allowed.contains($0) })
    }

    @Test("verifiers are unpredictable")
    func generatesDistinctVerifiers() {
        // A predictable verifier defeats the entire mechanism.
        let verifiers = Set((0..<50).map { _ in PKCE().verifier })
        #expect(verifiers.count == 50)
    }

    @Test("challenge is unpadded base64url of the SHA-256 digest")
    func computesChallengeCorrectly() {
        // The RFC 7636 appendix B test vector.
        let pkce = PKCE(verifier: "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk")
        #expect(pkce.challenge == "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM")

        // base64url, and unpadded — a `+`, `/` or `=` here is a rejected request.
        #expect(!pkce.challenge.contains("+"))
        #expect(!pkce.challenge.contains("/"))
        #expect(!pkce.challenge.contains("="))
        #expect(pkce.method == "S256")
    }

    // MARK: - Authorization request

    @Test("authorization URL carries every required parameter")
    func buildsAuthorizationURL() {
        let pkce = PKCE()
        let url = OAuthFlow.authorizationURL(Self.configuration, pkce: pkce, state: "state-abc")
        let items = URLComponents(url: url, resolvingAgainstBaseURL: false)!.queryItems ?? []

        func value(_ name: String) -> String? { items.first { $0.name == name }?.value }

        #expect(value("response_type") == "code")
        #expect(value("client_id") == "client-123")
        #expect(value("redirect_uri") == "http://localhost:8765/callback")
        #expect(value("code_challenge") == pkce.challenge)
        #expect(value("code_challenge_method") == "S256")
        #expect(value("state") == "state-abc")
        #expect(value("scope") == "read write")

        // The verifier is the secret half; sending it here would defeat PKCE.
        #expect(!url.absoluteString.contains(pkce.verifier))
    }

    // MARK: - Redirect handling

    @Test("a matching state yields the code")
    func extractsCode() {
        let url = URL(string: "http://localhost:8765/callback?code=abc123&state=state-abc")!
        #expect(OAuthFlow.code(fromRedirect: url, expectedState: "state-abc") == "abc123")
    }

    @Test("a mismatched state is rejected")
    func rejectsStateMismatch() {
        // Skipping this check is the classic authorization-code interception
        // bug: an attacker's redirect gets redeemed as the user's.
        let url = URL(string: "http://localhost:8765/callback?code=abc123&state=attacker")!
        #expect(OAuthFlow.code(fromRedirect: url, expectedState: "state-abc") == nil)
    }

    @Test("a redirect with no state is rejected")
    func rejectsMissingState() {
        let url = URL(string: "http://localhost:8765/callback?code=abc123")!
        #expect(OAuthFlow.code(fromRedirect: url, expectedState: "state-abc") == nil)
    }

    // MARK: - Token requests

    @Test("token exchange sends the verifier, form-encoded")
    func buildsTokenExchange() throws {
        let pkce = PKCE()
        let request = OAuthFlow.tokenExchange(Self.configuration, code: "code-xyz", pkce: pkce)

        #expect(request.httpMethod == "POST")
        // Token endpoints take form encoding, not JSON.
        #expect(request.value(forHTTPHeaderField: "content-type") == "application/x-www-form-urlencoded")

        let body = String(decoding: try #require(request.httpBody), as: UTF8.self)
        #expect(body.contains("grant_type=authorization_code"))
        #expect(body.contains("code=code-xyz"))
        #expect(body.contains("code_verifier=\(pkce.verifier)"))
    }

    @Test("refresh sends the refresh grant")
    func buildsRefresh() throws {
        let request = OAuthFlow.refresh(Self.configuration, refreshToken: "refresh-xyz")
        let body = String(decoding: try #require(request.httpBody), as: UTF8.self)

        #expect(body.contains("grant_type=refresh_token"))
        #expect(body.contains("refresh_token=refresh-xyz"))
    }

    // MARK: - Credentials

    @Test("a token response becomes a credential")
    func parsesTokenResponse() throws {
        let now = Date(timeIntervalSince1970: 1_000_000)
        let credential = try #require(OAuthCredential(tokenResponse: [
            "access_token": "at_1",
            "refresh_token": "rt_1",
            // Servers report a lifetime, not a deadline.
            "expires_in": 3600,
            "scope": "read write",
        ], now: now))

        #expect(credential.accessToken == "at_1")
        #expect(credential.refreshToken == "rt_1")
        #expect(credential.expiresAt == now.addingTimeInterval(3600))
        #expect(credential.scopes == ["read", "write"])
    }

    @Test("a response with no access token is not a credential")
    func rejectsMalformedTokenResponse() {
        #expect(OAuthCredential(tokenResponse: ["error": "invalid_grant"]) == nil)
    }

    @Test("refresh is triggered before expiry, not after")
    func refreshesWithLeeway() {
        let now = Date(timeIntervalSince1970: 1_000_000)

        // A token valid *now* can still expire mid-request, and a long
        // streaming response widens that window considerably.
        let expiringSoon = OAuthCredential(accessToken: "a", expiresAt: now.addingTimeInterval(60))
        #expect(expiringSoon.needsRefresh(now: now))

        let comfortable = OAuthCredential(accessToken: "a", expiresAt: now.addingTimeInterval(3600))
        #expect(!comfortable.needsRefresh(now: now))

        // No expiry reported means nothing to pre-empt.
        #expect(!OAuthCredential(accessToken: "a").needsRefresh(now: now))
    }

    // MARK: - Client integration

    @Test("an OAuth token travels as a bearer token, not an API key")
    func sendsOAuthAsBearer() throws {
        // Converting a working request from a key to a token is a *header*
        // change. Sending a token on `x-api-key` fails as an authentication
        // error that looks like a bad token.
        let client = try ManifoldClient(
            providerId: "anthropic",
            configuration: .init(authorization: .oauth(OAuthCredential(accessToken: "oat_1")))
        )
        let headers = client.authHeaders(wire: .anthropicMessages)

        #expect(headers["authorization"] == "Bearer oat_1")
        #expect(headers["x-api-key"] == nil)
        // Anthropic gates OAuth-authenticated requests behind this opt-in.
        #expect(headers["anthropic-beta"] == "oauth-2025-04-20")
        #expect(headers["anthropic-version"] == "2023-06-01")
    }

    @Test("an API key still travels as an API key")
    func sendsApiKeyAsKey() throws {
        let client = try ManifoldClient(
            providerId: "anthropic",
            configuration: .init(apiKey: "sk-1")
        )
        let headers = client.authHeaders(wire: .anthropicMessages)

        #expect(headers["x-api-key"] == "sk-1")
        #expect(headers["authorization"] == nil)
        // The OAuth opt-in must not leak onto key-authenticated requests.
        #expect(headers["anthropic-beta"] == nil)
    }

    @Test("the required version header survives an unauthenticated request")
    func keepsVersionHeaderWithoutAuth() throws {
        // A local Anthropic-compatible server needs no credential but still
        // requires the version header; omitting it is a 400.
        let client = try ManifoldClient(providerId: "anthropic", configuration: .init())
        #expect(client.authHeaders(wire: .anthropicMessages)["anthropic-version"] == "2023-06-01")
        #expect(client.authHeaders(wire: .openAICompletions).isEmpty)
    }
}
