import Foundation

/// Encodes ChatGPT Codex requests explicitly while reusing the Responses input
/// item and tool shapes that are genuinely identical.
public enum OpenAICodexResponsesRequest {
    public static let defaultInstructions = "You are a helpful assistant."
    static let promptCacheKeyMaxLength = 64

    public static func encode(
        _ options: CallOptions,
        model: ModelInfo? = nil,
        providerId: String = "openai-codex"
    ) -> EncodedRequest {
        let systemText = options.prompt
            .filter { $0.role == .system }
            .map(\.text)
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")

        var body: [String: JSONValue] = [
            "model": .string(options.model),
            // The ChatGPT backend rejects `store: true`.
            "store": .bool(false),
            // ChatGPT's Codex Responses transport is SSE-only.
            "stream": .bool(true),
            "instructions": .string(systemText.isEmpty ? defaultInstructions : systemText),
            "input": .array(OpenAIResponsesRequest.encodeInput(options.prompt)),
            "text": .object(["verbosity": .string("low")]),
            "include": .array([.string("reasoning.encrypted_content")]),
            "tool_choice": options.toolChoice.map(OpenAIResponsesRequest.encodeToolChoice)
                ?? .string("auto"),
            "parallel_tool_calls": .bool(true),
        ]
        var warnings: [Warning] = []

        if let maxTokens = options.maxOutputTokens {
            body["max_output_tokens"] = .number(Double(maxTokens))
        }
        if !options.tools.isEmpty {
            // Codex's Responses dialect uses an explicit null to request the
            // backend's strictness default unless the caller requested strict.
            body["tools"] = .array(OpenAIResponsesRequest.encodeTools(
                options.tools,
                defaultStrict: .null
            ))
        }
        if let temperature = options.temperature {
            if model?.supportsTemperature == false {
                warnings.append(Warning(
                    message: "\(options.model) does not accept temperature; dropped.",
                    setting: "temperature"
                ))
            } else {
                body["temperature"] = .number(temperature)
            }
        }
        if let topP = options.topP {
            body["top_p"] = .number(topP)
        }
        if !options.stopSequences.isEmpty {
            warnings.append(Warning(
                message: "The Codex Responses API has no stop-sequence parameter; dropped.",
                setting: "stopSequences"
            ))
        }

        if let thinking = options.thinking {
            let plan = thinking.plan(model: model, modelId: options.model)
            warnings += plan.warnings

            let effort: String? = if plan.enabled {
                plan.effort
            } else if plan.canDisable {
                plan.offEffort
            } else {
                nil
            }
            if let effort {
                body["reasoning"] = .object([
                    "effort": .string(effort),
                    "summary": .string("auto"),
                ])
            }
        }

        // Provider-specific escape-hatch values retain the existing policy:
        // they apply last and may override defaults deliberately.
        for (key, value) in options.options(for: providerId) {
            body[key] = value
        }
        if let key = body["prompt_cache_key"]?.stringValue {
            body["prompt_cache_key"] = .string(clampPromptCacheKey(key))
        }

        return EncodedRequest(body: .object(body), warnings: warnings)
    }

    /// The Codex cache/session identifier limit is measured in Unicode code
    /// points, matching JavaScript's `Array.from`, rather than UTF-8 bytes.
    static func clampPromptCacheKey(_ key: String) -> String {
        guard key.unicodeScalars.count > promptCacheKeyMaxLength else { return key }
        return String(key.unicodeScalars.prefix(promptCacheKeyMaxLength))
    }
}
