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
                CallOptions(model: "m", prompt: [.user("hi")], thinking: .level(.high)),
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

    @Test("turning thinking off renders into each vendor's own shape")
    func rendersThinkingOffPerDialect() {
        // The half of the problem that a "reasoning effort" field cannot
        // express: providers that think by default have to be told not to.
        func body(_ providerId: String) -> JSONValue {
            OpenAICompletionsRequest.encode(
                CallOptions(model: "m", prompt: [.user("hi")], thinking: .off),
                dialect: .forProvider(providerId)
            ).body
        }

        #expect(body("deepseek")["thinking"]?["type"]?.stringValue == "disabled")
        #expect(body("zai")["thinking"]?["type"]?.stringValue == "disabled")
        #expect(body("qwen")["enable_thinking"]?.boolValue == false)
        #expect(body("qwen")["thinking_budget"]?.intValue == 0)
        #expect(body("ollama")["chat_template_kwargs"]?["enable_thinking"]?.boolValue == false)
        #expect(body("openrouter")["reasoning"]?["enabled"]?.boolValue == false)
    }

    @Test("off is expressed as an effort only where the model has a word for it")
    func offEffortFollowsTheModelVocabulary() {
        // `reasoning_effort: "none"` is a 400 on a model whose vocabulary
        // stops at `low`, so absence of the field is the only safe off.
        let none = ModelInfo(
            id: "m",
            reasoning: .init(supported: true, default: true),
            reasoningOptions: [.init(type: "effort", values: ["none", "low", "high"])]
        )
        let bounded = ModelInfo(
            id: "m",
            reasoning: .init(supported: true, default: false),
            reasoningOptions: [.init(type: "effort", values: ["low", "medium", "high"])]
        )
        let options = CallOptions(model: "m", prompt: [.user("hi")], thinking: .off)

        #expect(
            OpenAICompletionsRequest.encode(options, model: none, dialect: .default)
                .body["reasoning_effort"]?.stringValue == "none"
        )
        #expect(
            OpenAICompletionsRequest.encode(options, model: bounded, dialect: .default)
                .body["reasoning_effort"] == nil
        )
    }

    @Test("a model that always thinks warns instead of sending a rejected disable")
    func warnsWhenThinkingCannotBeDisabled() {
        // Reasoning-only models exist; asking one to stop is a request the
        // provider would reject, so it is dropped and reported.
        let alwaysThinks = ModelInfo(
            id: "m",
            reasoning: .init(supported: true, default: true),
            reasoningOptions: [.init(type: "effort", values: ["low", "medium", "high"])]
        )

        let encoded = OpenAICompletionsRequest.encode(
            CallOptions(model: "m", prompt: [.user("hi")], thinking: .off),
            model: alwaysThinks,
            dialect: .forProvider("deepseek")
        )

        #expect(encoded.warnings.contains { $0.setting == "thinking" })
        #expect(encoded.body["thinking"] == nil)
    }

    @Test("a requested level is clamped to the model's own vocabulary")
    func clampsLevelToSupportedEfforts() {
        func effort(_ level: ThinkingLevel, values: [String]) -> String? {
            OpenAICompletionsRequest.encode(
                CallOptions(model: "m", prompt: [.user("hi")], thinking: .level(level)),
                model: ModelInfo(
                    id: "m",
                    reasoning: .init(supported: true),
                    reasoningOptions: [.init(type: "effort", values: values)]
                ),
                dialect: .default
            ).body["reasoning_effort"]?.stringValue
        }

        // `max` on a model that stops at `high` is `high`, not a 400.
        #expect(effort(.max, values: ["low", "medium", "high"]) == "high")
        #expect(effort(.minimal, values: ["low", "medium", "high"]) == "low")
        #expect(effort(.medium, values: ["high", "max"]) == "high")
        // Never `none`: asking to think harder must not turn thinking off.
        #expect(effort(.low, values: ["none", "medium", "high"]) == "medium")
    }

    @Test("a provider with no reasoning control warns instead of guessing")
    func warnsWhenReasoningUnsupported() {
        let encoded = OpenAICompletionsRequest.encode(
            CallOptions(model: "m", prompt: [.user("hi")], thinking: .level(.high)),
            dialect: .forProvider("minimax")
        )

        #expect(encoded.warnings.contains { $0.setting == "thinking" })
        #expect(encoded.body["reasoning_effort"] == nil)
        #expect(encoded.body["thinking"] == nil)
    }

    @Test("a provider-level toggle overrides the protocol's own spelling")
    func honoursProviderReasoningToggle() {
        // MiniMax speaks Anthropic but renames the values; the catalog carries
        // the rename, so no branch is needed here.
        let toggle = ProviderInfo.ReasoningToggle(
            field: "thinking.type", enabled: "adaptive", disabled: "disabled"
        )

        let on = AnthropicMessagesRequest.encode(
            CallOptions(model: "m", prompt: [.user("hi")], thinking: .on),
            reasoningToggle: toggle
        ).body
        #expect(on["thinking"]?["type"]?.stringValue == "adaptive")

        let off = AnthropicMessagesRequest.encode(
            CallOptions(model: "m", prompt: [.user("hi")], thinking: .off),
            reasoningToggle: toggle
        ).body
        #expect(off["thinking"]?["type"]?.stringValue == "disabled")
    }

    @Test("the catalog's reasoning toggles all decode")
    func decodesCatalogReasoningToggles() {
        let toggles = ProviderCatalog.all.compactMap(\.reasoningToggle)

        #expect(toggles.allSatisfy { !$0.field.isEmpty && $0.disabled != nil })
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

    // MARK: - Interleaved reasoning

    /// One assistant turn of the shape that breaks DeepSeek: thinking, then a
    /// tool call, then the result coming back.
    private static let toolTurn: Prompt = [
        .user("how did I sleep"),
        Message(role: .assistant, content: [
            .reasoning("the user wants last night, call sleep_summary", providerMetadata: nil),
            .toolCall(ToolCall(toolCallId: "c1", toolName: "sleep_summary", input: "{\"days\":7}")),
        ]),
        .toolResult(toolCallId: "c1", toolName: "sleep_summary", result: .string("5 nights")),
    ]

    @Test("DeepSeek gets the assistant's thinking replayed back")
    func replaysReasoningForDeepSeek() {
        // Without this the API answers 400 on every request after the first
        // tool call, which is every request an agent loop makes.
        let body = OpenAICompletionsRequest.encode(
            CallOptions(model: "deepseek-reasoner", prompt: Self.toolTurn),
            dialect: .forProvider("deepseek")
        ).body

        let assistant = body["messages"]?[1]
        #expect(assistant?["reasoning_content"]?.stringValue
            == "the user wants last night, call sleep_summary")
        #expect(assistant?["tool_calls"]?[0]?["id"]?.stringValue == "c1")
    }

    @Test("everyone else still gets no reasoning field")
    func dropsReasoningElsewhere() {
        // The mirror image of the DeepSeek 400: a provider that has never
        // heard of the field rejects a request that carries it.
        let body = OpenAICompletionsRequest.encode(
            CallOptions(model: "m", prompt: Self.toolTurn),
            dialect: .default
        ).body

        #expect(body["messages"]?[1]?["reasoning_content"] == nil)
    }

    @Test("an assistant turn with no thinking carries no empty field")
    func omitsFieldWithoutReasoning() {
        // Summaries and hand-written assistant turns have no thinking, and an
        // empty string is not the same claim as absence.
        let body = OpenAICompletionsRequest.encode(
            CallOptions(model: "deepseek-reasoner", prompt: [.user("hi"), .assistant("hello")]),
            dialect: .forProvider("deepseek")
        ).body

        #expect(body["messages"]?[1]?["reasoning_content"] == nil)
    }

    @Test("the model's own declaration wins over the dialect")
    func modelDeclarationWins() {
        // The dialect is the fallback for a model id the catalog predates.
        let model = ModelInfo(id: "custom", interleaved: .init(field: "thought_content"))
        let body = OpenAICompletionsRequest.encode(
            CallOptions(model: "custom", prompt: Self.toolTurn),
            model: model,
            dialect: .forProvider("deepseek")
        ).body

        #expect(body["messages"]?[1]?["thought_content"]?.stringValue != nil)
        #expect(body["messages"]?[1]?["reasoning_content"] == nil)
    }

    @Test("the catalog's DeepSeek thinking models declare the requirement")
    func catalogDeclaresInterleaved() {
        for id in ["deepseek-v4-flash", "deepseek-v4-pro"] {
            let model = ProviderCatalog.model(id, provider: "deepseek")?.1
            #expect(model?.interleavedReasoningField == "reasoning_content")
        }
    }

    @Test("the client selects the dialect from the provider")
    func clientAppliesDialect() throws {
        // Selection is per provider, not per protocol — that is the whole
        // point of the layer.
        let deepseek = try AIClient(providerId: "deepseek", configuration: .init(apiKey: "k"))
        let prepared = try deepseek.prepare(CallOptions(
            model: "deepseek-v4-flash",
            prompt: [.user("hi")],
            thinking: .level(.high)
        ))
        let body = deepseek.encode(prepared).body

        #expect(prepared.wire == .openAICompletions)
        #expect(body["thinking"]?["type"]?.stringValue == "enabled")
    }
}
