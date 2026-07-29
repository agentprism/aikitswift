import Foundation
import Testing

@testable import Manifold

/// Exercises request construction without touching the network.
@Suite("Client")
struct ClientTests {

    private func client(_ providerId: String, key: String = "sk-test") throws -> ManifoldClient {
        try ManifoldClient(providerId: providerId, configuration: .init(apiKey: key))
    }

    @Test("unknown providers fail with a helpful error")
    func rejectsUnknownProvider() {
        #expect(throws: ManifoldError.self) {
            try ManifoldClient(providerId: "not-a-real-provider", configuration: .init())
        }
    }

    @Test("endpoints resolve per protocol")
    func resolvesEndpoints() throws {
        let base = URL(string: "https://example.com")!
        let subject = try client("anthropic")

        #expect(
            subject.endpoint(wire: .anthropicMessages, base: base, model: "m").path == "/v1/messages"
        )
        #expect(
            subject.endpoint(wire: .openAICompletions, base: base, model: "m").path == "/v1/chat/completions"
        )
        #expect(
            subject.endpoint(wire: .openAIResponses, base: base, model: "m").path == "/v1/responses"
        )
    }

    @Test("a base URL that already has a version prefix is not doubled")
    func doesNotDoubleVersionPrefix() throws {
        // Catalog base URLs are inconsistent about including `/v1`; appending
        // blindly produces `/v1/v1/chat/completions` and a 404.
        let subject = try client("openai")
        let versioned = URL(string: "https://example.com/v1")!

        let url = subject.endpoint(wire: .openAICompletions, base: versioned, model: "m")
        #expect(url.path == "/v1/chat/completions")
        #expect(!url.path.contains("v1/v1"))
    }

    @Test("Gemini opts into SSE explicitly")
    func geminiRequestsSSE() throws {
        // Without `alt=sse` the API streams a JSON array instead of events,
        // and the SSE parser sees nothing at all.
        let subject = try client("google")
        let url = subject.endpoint(
            wire: .googleGenerativeAI,
            base: URL(string: "https://generativelanguage.googleapis.com")!,
            model: "gemini-3-pro"
        )

        #expect(url.absoluteString.contains("gemini-3-pro:streamGenerateContent"))
        #expect(url.absoluteString.contains("alt=sse"))
    }

    @Test("auth headers differ per protocol")
    func usesCorrectAuthHeaders() throws {
        let subject = try client("anthropic")

        let anthropic = subject.authHeaders(wire: .anthropicMessages)
        #expect(anthropic["x-api-key"] == "sk-test")
        // Required on every Anthropic request; omitting it is a 400.
        #expect(anthropic["anthropic-version"] == "2023-06-01")

        #expect(subject.authHeaders(wire: .openAICompletions)["authorization"] == "Bearer sk-test")
        #expect(subject.authHeaders(wire: .googleGenerativeAI)["x-goog-api-key"] == "sk-test")
    }

    @Test("no key means no auth headers")
    func omitsAuthWithoutKey() throws {
        // Local runtimes and gateways often need none.
        let subject = try ManifoldClient(providerId: "anthropic", configuration: .init())
        #expect(subject.authHeaders(wire: .anthropicMessages).isEmpty)
    }

    @Test("requests carry a JSON body and streaming headers")
    func buildsRequest() throws {
        let subject = try client("anthropic")
        let encoded = AnthropicMessagesRequest.encode(
            CallOptions(model: "claude-opus-4-8", prompt: [.user("hi")])
        )
        let request = try subject.makeRequest(
            wire: .anthropicMessages,
            options: CallOptions(model: "claude-opus-4-8", prompt: [.user("hi")]),
            body: encoded.body
        )

        #expect(request.httpMethod == "POST")
        #expect(request.value(forHTTPHeaderField: "content-type") == "application/json")
        #expect(request.value(forHTTPHeaderField: "accept") == "text/event-stream")
        #expect(request.httpBody != nil)

        let body = try JSONValue.decode(from: request.httpBody!)
        #expect(body["stream"]?.boolValue == true)
    }

    @Test("extra headers are applied last so they can override")
    func extraHeadersOverride() throws {
        let subject = try ManifoldClient(
            providerId: "anthropic",
            configuration: .init(apiKey: "sk-test", extraHeaders: ["anthropic-version": "2099-01-01"])
        )
        let request = try subject.makeRequest(
            wire: .anthropicMessages,
            options: CallOptions(model: "m", prompt: [.user("hi")]),
            body: .object([:])
        )

        #expect(request.value(forHTTPHeaderField: "anthropic-version") == "2099-01-01")
    }

    @Test("a configured base URL overrides the catalog")
    func configuredBaseURLWins() throws {
        // How a local or proxied endpoint is pointed at a catalog provider.
        let subject = try ManifoldClient(
            providerId: "openai",
            configuration: .init(apiKey: "k", baseURL: URL(string: "http://localhost:1234")!)
        )
        let request = try subject.makeRequest(
            wire: .openAICompletions,
            options: CallOptions(model: "m", prompt: [.user("hi")]),
            body: .object([:])
        )

        #expect(request.url?.host() == "localhost")
        #expect(request.url?.port == 1234)
    }

    @Test("the encoder is chosen by protocol, not by provider")
    func selectsEncoderByProtocol() throws {
        // DeepSeek speaks Chat Completions, so it must produce a Completions
        // body — this is the whole point of the protocol/provider split.
        let subject = try client("deepseek")
        let body = subject.encode(
            CallOptions(model: "deepseek-v4-flash", prompt: [.system("s"), .user("hi")]),
            wire: .openAICompletions,
            model: nil
        ).body

        #expect(body["messages"]?[0]?["role"]?.stringValue == "system")
        #expect(body["stream_options"] != nil)
    }
}
