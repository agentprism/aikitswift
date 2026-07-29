import Foundation
import Testing

@testable import AIKit

/// "Thirty-eight providers speak Chat Completions" is a simplification. These
/// tests pin the places where that stops being true.
@Suite("Chat Completions dialects")
struct DialectTests {

    private static let options = CallOptions(
        model: "m",
        prompt: [.user("hi")],
        maxOutputTokens: 100,
        tools: [ToolDefinition(name: "t", inputSchema: ["type": "object"], strict: true)]
    )

    @Test("an unknown provider gets the conservative baseline")
    func fallsBackToBaseline() {
        // A provider added to the catalog upstream must work before anyone has
        // characterized it here.
        let dialect = CompletionsDialect.forProvider("some-new-provider")
        #expect(dialect == .default)
        #expect(dialect.maxTokensField == .maxTokens)
        #expect(dialect.supportsUsageInStreaming)
    }

    @Test("known providers resolve to their dialect")
    func resolvesKnownProviders() {
        #expect(CompletionsDialect.forProvider("openai").maxTokensField == .maxCompletionTokens)
        #expect(CompletionsDialect.forProvider("deepseek").thinkingFormat == .deepseek)
        #expect(CompletionsDialect.forProvider("openrouter").thinkingFormat == .openrouter)
        #expect(CompletionsDialect.forProvider("zai").thinkingFormat == .zai)
        #expect(!CompletionsDialect.forProvider("zhipuai").supportsStrict)
        #expect(CompletionsDialect.forProvider("ollama").thinkingFormat == .chatTemplate)
    }

    @Test("suffixed provider ids inherit the base dialect")
    func matchesSuffixedIds() {
        // The catalog is full of `-coding-plan`, `-cn` and `-ams` variants of
        // the same upstream API.
        #expect(CompletionsDialect.forProvider("zai-coding-plan").thinkingFormat == .zai)
        #expect(CompletionsDialect.forProvider("zhipuai-coding-plan").supportsStrict == false)
        #expect(CompletionsDialect.forProvider("minimax-cn").thinkingFormat == .unsupported)
    }

    @Test("usage opt-in is omitted where it is rejected")
    func omitsUsageOptInWhereUnsupported() {
        // The one flag dangerous in both directions: sending it where it is
        // rejected is a 400; omitting it where supported loses all token
        // counts silently.
        let supported = OpenAICompletionsRequest.encode(Self.options, dialect: .default).body
        #expect(supported["stream_options"]?["include_usage"]?.boolValue == true)

        let unsupported = OpenAICompletionsRequest.encode(
            Self.options,
            dialect: .forProvider("ollama")
        ).body
        #expect(unsupported["stream_options"] == nil)
    }

    @Test("the token cap field follows the dialect")
    func picksTokenCapField() {
        let openai = OpenAICompletionsRequest.encode(
            Self.options, dialect: .forProvider("openai")
        ).body
        #expect(openai["max_completion_tokens"]?.intValue == 100)
        #expect(openai["max_tokens"] == nil)

        let generic = OpenAICompletionsRequest.encode(Self.options, dialect: .default).body
        #expect(generic["max_tokens"]?.intValue == 100)
    }

    @Test("a reasoning model overrides the dialect's token cap field")
    func reasoningModelForcesCompletionTokens() {
        // The rename came with reasoning models, so model capability wins over
        // the provider-level default.
        let body = OpenAICompletionsRequest.encode(
            Self.options,
            model: ModelInfo(id: "m", reasoning: .init(supported: true)),
            dialect: .default
        ).body

        #expect(body["max_completion_tokens"]?.intValue == 100)
    }

    @Test("strict is dropped where it is not implemented")
    func dropsStrictWhereUnsupported() {
        // Sending `strict` to a provider that does not implement it is a 400,
        // which would fail the whole request over an optional hint.
        let strict = OpenAICompletionsRequest.encode(Self.options, dialect: .default).body
        #expect(strict["tools"]?[0]?["function"]?["strict"]?.boolValue == true)

        let lenient = OpenAICompletionsRequest.encode(
            Self.options, dialect: .forProvider("zhipuai")
        ).body
        #expect(lenient["tools"]?[0]?["function"]?["strict"] == nil)
    }

    @Test("tool results gain a name where one is required")
    func addsToolResultName() {
        let prompt: Prompt = [.toolResult(
            toolCallId: "call_1", toolName: "get_weather", result: "18C"
        )]
        let options = CallOptions(model: "m", prompt: prompt)

        let baseline = OpenAICompletionsRequest.encode(options, dialect: .default).body
        #expect(baseline["messages"]?[0]?["name"] == nil)

        var strict = CompletionsDialect.default
        strict.requiresToolResultName = true
        let named = OpenAICompletionsRequest.encode(options, dialect: strict).body
        #expect(named["messages"]?[0]?["name"]?.stringValue == "get_weather")
    }

    @Test("reasoning renders into each vendor's own shape")
    func rendersReasoningPerDialect() {
        func body(_ providerId: String) -> JSONValue {
            OpenAICompletionsRequest.encode(
                CallOptions(model: "m", prompt: [.user("hi")], reasoningEffort: "high"),
                dialect: .forProvider(providerId)
            ).body
        }

        // Seven incompatible shapes for one idea.
        #expect(body("openai")["reasoning_effort"]?.stringValue == "high")
        #expect(body("openrouter")["reasoning"]?["effort"]?.stringValue == "high")
        #expect(body("deepseek")["thinking"]?["type"]?.stringValue == "enabled")
        #expect(body("zai")["thinking"]?["type"]?.stringValue == "enabled")
        #expect(body("qwen")["enable_thinking"]?.boolValue == true)
        #expect(body("ollama")["chat_template_kwargs"]?["enable_thinking"]?.boolValue == true)
    }

    @Test("a provider with no reasoning control warns instead of guessing")
    func warnsWhenReasoningUnsupported() {
        let encoded = OpenAICompletionsRequest.encode(
            CallOptions(model: "m", prompt: [.user("hi")], reasoningEffort: "high"),
            dialect: .forProvider("minimax")
        )

        #expect(encoded.warnings.contains { $0.setting == "reasoningEffort" })
        #expect(encoded.body["reasoning_effort"] == nil)
        #expect(encoded.body["thinking"] == nil)
    }

    @Test("no reasoning request means no reasoning field")
    func omitsReasoningWhenUnset() {
        let body = OpenAICompletionsRequest.encode(
            CallOptions(model: "m", prompt: [.user("hi")]),
            dialect: .forProvider("deepseek")
        ).body

        #expect(body["thinking"] == nil)
        #expect(body["reasoning_effort"] == nil)
    }

    @Test("every catalog provider on this protocol resolves a dialect")
    func coversTheCatalog() {
        // Not an assertion that each is *characterized* — only that none
        // crashes or resolves to nothing.
        for provider in ProviderCatalog.providers(speaking: .openAICompletions) {
            let dialect = CompletionsDialect.forProvider(provider.id)
            #expect(CompletionsDialect.ThinkingFormat.allCases.contains(dialect.thinkingFormat))
        }
    }

    @Test("the client selects the dialect from the provider")
    func clientAppliesDialect() throws {
        // Selection is per provider, not per protocol — that is the whole
        // point of the layer.
        let deepseek = try AIClient(providerId: "deepseek", configuration: .init(apiKey: "k"))
        let body = deepseek.encode(
            CallOptions(model: "deepseek-v4-flash", prompt: [.user("hi")], reasoningEffort: "high"),
            wire: .openAICompletions,
            model: nil
        ).body

        #expect(body["thinking"]?["type"]?.stringValue == "enabled")
    }
}
