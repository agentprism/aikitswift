import Foundation
import Testing

@testable import AIKit

/// Thinking is the one setting where "say nothing" and "say no" differ: several
/// providers reason by default, so an unset field buys tokens the caller may
/// not want. These pin both directions, per protocol.
@Suite("Thinking")
struct ThinkingTests {

    private static func options(_ thinking: Thinking?, model: String = "m") -> CallOptions {
        CallOptions(model: model, prompt: [.user("hi")], thinking: thinking)
    }

    // MARK: - Anthropic

    @Test("newer Anthropic models take adaptive thinking with a separate effort")
    func anthropicAdaptiveWithEffort() throws {
        let model = try #require(ProviderCatalog.model("claude-opus-4-8", provider: "anthropic")?.1)

        let body = AnthropicMessagesRequest.encode(
            Self.options(.level(.high), model: "claude-opus-4-8"), model: model
        ).body

        #expect(body["thinking"]?["type"]?.stringValue == "adaptive")
        // Effort is nested under output_config, not top-level.
        #expect(body["output_config"]?["effort"]?.stringValue == "high")
        #expect(body["thinking"]?["budget_tokens"] == nil)
    }

    @Test("older Anthropic models take a token budget instead")
    func anthropicBudgetTokens() throws {
        let model = try #require(ProviderCatalog.model("claude-opus-4-1", provider: "anthropic")?.1)

        let body = AnthropicMessagesRequest.encode(
            Self.options(.level(.high), model: "claude-opus-4-1"), model: model
        ).body

        #expect(body["thinking"]?["type"]?.stringValue == "enabled")
        let budget = try #require(body["thinking"]?["budget_tokens"]?.intValue)
        #expect(budget >= 1024)
        // The budget comes out of the same allowance as the answer, so it must
        // leave room for one.
        #expect(budget < (body["max_tokens"]?.intValue ?? 0))
    }

    @Test("a budget larger than the output cap is brought back under it")
    func anthropicBudgetStaysUnderMaxTokens() {
        var options = Self.options(.budget(tokens: 100_000))
        options.maxOutputTokens = 8000

        let body = AnthropicMessagesRequest.encode(options).body

        #expect(body["thinking"]?["budget_tokens"]?.intValue == 8000 - 1024)
    }

    @Test("Anthropic thinking is disabled explicitly, not by omission")
    func anthropicDisables() {
        let body = AnthropicMessagesRequest.encode(Self.options(.off)).body

        #expect(body["thinking"]?["type"]?.stringValue == "disabled")
    }

    @Test("a model that always thinks is not sent a disable it would reject")
    func anthropicAlwaysThinkingModel() throws {
        let model = try #require(ProviderCatalog.model("claude-fable-5", provider: "anthropic")?.1)

        let encoded = AnthropicMessagesRequest.encode(
            Self.options(.off, model: "claude-fable-5"), model: model
        )

        #expect(encoded.body["thinking"] == nil)
        #expect(encoded.warnings.contains { $0.setting == "thinking" })
    }

    @Test("saying nothing about thinking sends no thinking field")
    func anthropicOmitsWhenUnset() {
        #expect(AnthropicMessagesRequest.encode(Self.options(nil)).body["thinking"] == nil)
    }

    // MARK: - Google

    @Test("Gemini 3 takes a named thinking level")
    func geminiThinkingLevel() throws {
        let model = try #require(ProviderCatalog.model("gemini-3.5-flash", provider: "google")?.1)

        let body = GoogleGenerativeAIRequest.encode(
            Self.options(.level(.low), model: "gemini-3.5-flash"), model: model
        ).body

        // Nested inside generationConfig, not top-level.
        #expect(body["generationConfig"]?["thinkingConfig"]?["thinkingLevel"]?.stringValue == "low")
    }

    @Test("Gemini 2.5 takes a token budget, and zero is off")
    func geminiThinkingBudget() throws {
        let model = try #require(ProviderCatalog.model("gemini-2.5-flash", provider: "google")?.1)

        let off = GoogleGenerativeAIRequest.encode(
            Self.options(.off, model: "gemini-2.5-flash"), model: model
        ).body
        #expect(off["generationConfig"]?["thinkingConfig"]?["thinkingBudget"]?.intValue == 0)

        let on = GoogleGenerativeAIRequest.encode(
            Self.options(.level(.high), model: "gemini-2.5-flash"), model: model
        ).body
        let budget = try #require(on["generationConfig"]?["thinkingConfig"]?["thinkingBudget"]?.intValue)
        #expect((0...24576).contains(budget))
    }

    @Test("a Gemini model with no silence setting is taken to its floor, and says so")
    func geminiCannotFullyDisable() throws {
        // Gemini 3 Flash bottoms out at `minimal` — the request is honoured as
        // closely as the model allows, and the caller is told the difference.
        let model = try #require(ProviderCatalog.model("gemini-3.5-flash", provider: "google")?.1)

        let encoded = GoogleGenerativeAIRequest.encode(
            Self.options(.off, model: "gemini-3.5-flash"), model: model
        )

        #expect(
            encoded.body["generationConfig"]?["thinkingConfig"]?["thinkingLevel"]?.stringValue
                == "minimal"
        )
        #expect(encoded.warnings.contains { $0.setting == "thinking" })
    }

    @Test("an unset thinking setting leaves Gemini's config alone")
    func geminiOmitsWhenUnset() {
        let body = GoogleGenerativeAIRequest.encode(Self.options(nil)).body

        #expect(body["generationConfig"]?["thinkingConfig"] == nil)
    }

    // MARK: - OpenAI Responses

    @Test("the Responses API carries both directions in one nested field")
    func responsesReasoning() {
        let model = ModelInfo(
            id: "m",
            reasoning: .init(supported: true, default: true),
            reasoningOptions: [.init(type: "effort", values: ["none", "low", "medium", "high"])]
        )

        let on = OpenAIResponsesRequest.encode(Self.options(.level(.medium)), model: model).body
        #expect(on["reasoning"]?["effort"]?.stringValue == "medium")

        let off = OpenAIResponsesRequest.encode(Self.options(.off), model: model).body
        #expect(off["reasoning"]?["effort"]?.stringValue == "none")
    }

    // MARK: - Levels

    @Test("a settings string parses into a thinking request")
    func parsesSettingStrings() {
        #expect(Thinking(setting: "off") == .off)
        #expect(Thinking(setting: "none") == .off)
        #expect(Thinking(setting: "auto") == .on)
        #expect(Thinking(setting: "High") == .level(.high))
        #expect(Thinking(setting: "banana") == nil)
    }

    @Test("a zero budget means off")
    func zeroBudgetIsOff() {
        #expect(Thinking.budget(tokens: 0).isOff)

        let body = AnthropicMessagesRequest.encode(Self.options(.budget(tokens: 0))).body
        #expect(body["thinking"]?["type"]?.stringValue == "disabled")
    }

    @Test("levels are ordered")
    func levelsAreOrdered() {
        #expect(ThinkingLevel.minimal < .low)
        #expect(ThinkingLevel.high < .xhigh)
        #expect(ThinkingLevel.xhigh < .max)
    }

    // MARK: - Capabilities

    @Test("every catalog model that reasons resolves a capability")
    func capabilitiesCoverTheCatalog() {
        // Not an assertion about any one model — only that the derivation
        // never contradicts itself on real data.
        for provider in ProviderCatalog.all {
            for model in provider.models ?? [] {
                let capability = model.thinkingCapability
                if let range = capability.budgetRange {
                    #expect(range.lowerBound <= range.upperBound)
                }
                if capability.hasToggle {
                    #expect(capability.canDisable)
                }
            }
        }
    }

    @Test("a model that reasons by default and offers a toggle can be quieted")
    func toggleMakesAModelDisablable() throws {
        let model = try #require(ProviderCatalog.model("deepseek-v4-flash", provider: "deepseek")?.1)

        #expect(model.reasoning?.default == true)
        #expect(model.thinkingCapability.canDisable)

        let body = OpenAICompletionsRequest.encode(
            Self.options(.off, model: "deepseek-v4-flash"),
            model: model,
            dialect: .forProvider("deepseek")
        ).body

        #expect(body["thinking"]?["type"]?.stringValue == "disabled")
    }
}
