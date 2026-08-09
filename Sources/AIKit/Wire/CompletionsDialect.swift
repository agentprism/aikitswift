import Foundation

/// Per-provider deviations from the OpenAI Chat Completions API.
///
/// "Thirty-eight providers speak Chat Completions" is a useful simplification,
/// not the truth. They speak thirty-eight dialects of it: some renamed the
/// token cap, some reject `strict`, some require a `name` on tool results, and
/// reasoning alone has a dozen incompatible request shapes.
///
/// Encoding those differences as data rather than as branches keeps one mapper
/// and one encoder serving all of them. This is the same approach pi-ai arrived
/// at; the alternative — a package per provider — is what makes the JavaScript
/// SDKs so large.
///
/// A wrong dialect usually surfaces as a 400 from the provider rather than as
/// anything subtle, which is the good case. `supportsUsageInStreaming` is the
/// exception: get it wrong in one direction and the request fails, in the other
/// and token counts silently vanish.
public struct CompletionsDialect: Sendable, Hashable {

    /// Which field carries the output token cap.
    public enum MaxTokensField: String, Sendable, Hashable {
        case maxTokens = "max_tokens"
        /// Reasoning models rejected `max_tokens`; the budget also covers
        /// reasoning here.
        case maxCompletionTokens = "max_completion_tokens"
    }

    /// How a provider expresses "think harder" — and, just as importantly,
    /// "don't think".
    ///
    /// The single field with the most divergence in the entire ecosystem. Note
    /// that every format below has *two* shapes, not one: a provider that
    /// defaults thinking on can only be quieted by sending the off shape, so a
    /// format that only knows how to enable is half a format.
    public enum ThinkingFormat: String, Sendable, Hashable, CaseIterable {
        /// `reasoning_effort: "high"` / `"none"` — OpenAI and most followers.
        case openai
        /// `reasoning: { effort: "high" }` / `{ enabled: false }`.
        case openrouter
        /// `thinking: { type: "enabled" | "disabled" }`, plus effort where
        /// supported.
        case deepseek
        /// `reasoning: { enabled: true | false }`, plus effort where supported.
        case together
        /// `thinking: { type: "enabled" | "disabled" }`.
        case zai
        /// Top-level `enable_thinking: true | false`, plus `thinking_budget`.
        case qwen
        /// `chat_template_kwargs: { enable_thinking: true | false }` —
        /// self-hosted runtimes that pass the flag through to the chat
        /// template.
        case chatTemplate
        /// The provider has no reasoning control.
        case unsupported
    }

    public var maxTokensField: MaxTokensField
    /// Whether tool definitions accept `strict`.
    public var supportsStrict: Bool
    /// Whether the `developer` role is accepted in place of `system`.
    public var supportsDeveloperRole: Bool
    /// Whether tool result messages must repeat the tool's `name`.
    public var requiresToolResultName: Bool
    /// Whether `stream_options: { include_usage: true }` is accepted.
    ///
    /// The one flag that is dangerous in both directions: sending it to a
    /// provider that rejects it is a 400, and omitting it where it is supported
    /// means the response carries no usage at all.
    public var supportsUsageInStreaming: Bool
    /// Whether the `store` field is accepted.
    public var supportsStore: Bool
    public var thinkingFormat: ThinkingFormat
    /// The field an assistant message's own thinking must be replayed in, for
    /// the providers that require it. Nil — the overwhelming majority — means
    /// reasoning is write-only and is dropped from the history.
    ///
    /// This is the fallback for a model the catalog does not describe; the
    /// model's own `interleaved.field` wins when there is one. Getting it
    /// wrong in either direction is a 400: DeepSeek rejects a thinking-mode
    /// tool request that omits `reasoning_content`, and providers that have
    /// never heard of the field reject a request that carries it.
    public var interleavedReasoningField: String?

    public init(
        maxTokensField: MaxTokensField = .maxTokens,
        supportsStrict: Bool = true,
        supportsDeveloperRole: Bool = false,
        requiresToolResultName: Bool = false,
        supportsUsageInStreaming: Bool = true,
        supportsStore: Bool = false,
        thinkingFormat: ThinkingFormat = .openai,
        interleavedReasoningField: String? = nil
    ) {
        self.maxTokensField = maxTokensField
        self.supportsStrict = supportsStrict
        self.supportsDeveloperRole = supportsDeveloperRole
        self.requiresToolResultName = requiresToolResultName
        self.supportsUsageInStreaming = supportsUsageInStreaming
        self.supportsStore = supportsStore
        self.thinkingFormat = thinkingFormat
        self.interleavedReasoningField = interleavedReasoningField
    }

    /// The conservative baseline: what a generic OpenAI-compatible endpoint
    /// can be assumed to accept.
    public static let `default` = CompletionsDialect()

    /// The dialect for a catalog provider id.
    ///
    /// Unknown ids fall back to the baseline, so a provider added upstream
    /// works before anyone has characterized it here.
    public static func forProvider(_ id: String) -> CompletionsDialect {
        for (prefixes, dialect) in table {
            if prefixes.contains(where: { id == $0 || id.hasPrefix("\($0)-") }) {
                return dialect
            }
        }
        return .default
    }

    private static let table: [([String], CompletionsDialect)] = [
        (["openai", "azure", "openai-codex"], CompletionsDialect(
            maxTokensField: .maxCompletionTokens,
            supportsDeveloperRole: true,
            supportsStore: true
        )),
        (["deepseek"], CompletionsDialect(
            thinkingFormat: .deepseek,
            // Thinking is on by default here, and a thinking-mode request that
            // carries `tools` must replay every intermediate assistant's
            // `reasoning_content` or the API answers 400. An agent loop hits
            // that on its second request — the one right after the first tool
            // call — so the whole provider is unusable without this.
            interleavedReasoningField: "reasoning_content"
        )),
        (["zai", "zhipuai"], CompletionsDialect(
            // Rejects `strict` on tool definitions.
            supportsStrict: false,
            thinkingFormat: .zai
        )),
        (["moonshot", "kimi"], CompletionsDialect(supportsStrict: false)),
        (["openrouter"], CompletionsDialect(thinkingFormat: .openrouter)),
        (["qwen", "bailian"], CompletionsDialect(
            supportsStrict: false,
            thinkingFormat: .qwen
        )),
        (["minimax"], CompletionsDialect(supportsStrict: false, thinkingFormat: .unsupported)),
        (["ollama", "lm-studio"], CompletionsDialect(
            supportsStrict: false,
            // Local runtimes commonly reject the field outright.
            supportsUsageInStreaming: false,
            thinkingFormat: .chatTemplate
        )),
        (["siliconflow", "modelscope", "ppinfra", "novita", "qiniu"], CompletionsDialect(
            supportsStrict: false,
            thinkingFormat: .chatTemplate
        )),
        (["groq", "cerebras", "fireworks", "nvidia", "upstage", "stepfun"], CompletionsDialect(
            supportsStrict: false
        )),
        (["xai"], CompletionsDialect()),
    ]

    // MARK: - Application

    /// Renders a resolved thinking request in whichever shape this provider
    /// expects, and returns anything that could not be expressed.
    ///
    /// - Parameter toggle: a provider-level override from the catalog, which
    ///   wins over the dialect's own spelling when present.
    func applyThinking(
        _ plan: ThinkingPlan,
        toggle: ProviderInfo.ReasoningToggle? = nil,
        to body: inout [String: JSONValue]
    ) -> [Warning] {
        var warnings = plan.warnings

        if let toggle, let value = plan.enabled ? toggle.enabled : toggle.disabled {
            guard plan.enabled || plan.canDisable else { return warnings }
            toggle.apply(value, to: &body)
            if plan.enabled, let effort = plan.effort {
                body["reasoning_effort"] = .string(effort)
            }
            return warnings
        }

        guard thinkingFormat != .unsupported else {
            warnings.append(Warning(
                message: "This provider has no reasoning-control parameter; the thinking setting was dropped.",
                setting: "thinking"
            ))
            return warnings
        }

        guard plan.enabled else {
            // A model that cannot be quieted has already produced a warning;
            // sending a disable it rejects would only turn that into a 400.
            guard plan.canDisable else { return warnings }

            switch thinkingFormat {
            case .openai:
                // `reasoning_effort: "none"` exists only where the catalog says
                // so; elsewhere omitting the field *is* the off switch.
                if let off = plan.offEffort {
                    body["reasoning_effort"] = .string(off)
                }
            case .openrouter:
                body["reasoning"] = .object(["enabled": .bool(false)])
            case .deepseek, .zai:
                body["thinking"] = .object(["type": .string("disabled")])
            case .together:
                body["reasoning"] = .object(["enabled": .bool(false)])
            case .qwen:
                body["enable_thinking"] = .bool(false)
                body["thinking_budget"] = .number(0)
            case .chatTemplate:
                body["chat_template_kwargs"] = .object(["enable_thinking": .bool(false)])
            case .unsupported:
                break
            }
            return warnings
        }

        switch thinkingFormat {
        case .openai:
            if let effort = plan.effort { body["reasoning_effort"] = .string(effort) }
        case .openrouter:
            var reasoning: [String: JSONValue] = ["enabled": .bool(true)]
            if let effort = plan.effort { reasoning["effort"] = .string(effort) }
            body["reasoning"] = .object(reasoning)
        case .deepseek:
            body["thinking"] = .object(["type": .string("enabled")])
            if let effort = plan.effort { body["reasoning_effort"] = .string(effort) }
        case .together:
            body["reasoning"] = .object(["enabled": .bool(true)])
            if let effort = plan.effort { body["reasoning_effort"] = .string(effort) }
        case .zai:
            body["thinking"] = .object(["type": .string("enabled")])
        case .qwen:
            body["enable_thinking"] = .bool(true)
            if let budget = plan.budgetTokens { body["thinking_budget"] = .number(Double(budget)) }
        case .chatTemplate:
            body["chat_template_kwargs"] = .object(["enable_thinking": .bool(true)])
        case .unsupported:
            break
        }

        return warnings
    }
}
