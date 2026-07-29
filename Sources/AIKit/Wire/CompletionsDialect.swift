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

    /// How a provider expresses "think harder".
    ///
    /// The single field with the most divergence in the entire ecosystem.
    public enum ThinkingFormat: String, Sendable, Hashable, CaseIterable {
        /// `reasoning_effort: "high"` — OpenAI and most followers.
        case openai
        /// `reasoning: { effort: "high" }`.
        case openrouter
        /// `thinking: { type: "enabled" }`, plus effort where supported.
        case deepseek
        /// `reasoning: { enabled: true }`, plus effort where supported.
        case together
        /// `thinking: { type: "enabled" }`.
        case zai
        /// Top-level `enable_thinking: true`.
        case qwen
        /// `chat_template_kwargs: { enable_thinking: true }` — self-hosted
        /// runtimes that pass the flag through to the chat template.
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

    public init(
        maxTokensField: MaxTokensField = .maxTokens,
        supportsStrict: Bool = true,
        supportsDeveloperRole: Bool = false,
        requiresToolResultName: Bool = false,
        supportsUsageInStreaming: Bool = true,
        supportsStore: Bool = false,
        thinkingFormat: ThinkingFormat = .openai
    ) {
        self.maxTokensField = maxTokensField
        self.supportsStrict = supportsStrict
        self.supportsDeveloperRole = supportsDeveloperRole
        self.requiresToolResultName = requiresToolResultName
        self.supportsUsageInStreaming = supportsUsageInStreaming
        self.supportsStore = supportsStore
        self.thinkingFormat = thinkingFormat
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
        (["deepseek"], CompletionsDialect(thinkingFormat: .deepseek)),
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

    /// Adds the reasoning request in whichever shape this provider expects.
    ///
    /// - Parameter effort: `low`, `medium`, `high`, or similar.
    func applyThinking(effort: String?, enabled: Bool, to body: inout [String: JSONValue]) {
        guard enabled else { return }

        switch thinkingFormat {
        case .openai:
            if let effort { body["reasoning_effort"] = .string(effort) }
        case .openrouter:
            var reasoning: [String: JSONValue] = [:]
            if let effort { reasoning["effort"] = .string(effort) }
            body["reasoning"] = .object(reasoning)
        case .deepseek:
            body["thinking"] = .object(["type": .string("enabled")])
            if let effort { body["reasoning_effort"] = .string(effort) }
        case .together:
            body["reasoning"] = .object(["enabled": .bool(true)])
            if let effort { body["reasoning_effort"] = .string(effort) }
        case .zai:
            body["thinking"] = .object(["type": .string("enabled")])
        case .qwen:
            body["enable_thinking"] = .bool(true)
        case .chatTemplate:
            body["chat_template_kwargs"] = .object(["enable_thinking": .bool(true)])
        case .unsupported:
            break
        }
    }
}
