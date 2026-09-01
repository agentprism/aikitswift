import Foundation

/// Encodes ``CallOptions`` into an Anthropic Messages request body.
public enum AnthropicMessagesRequest {

    /// - Parameters:
    ///   - model: catalog metadata for the target model, when known. Used to
    ///     drop settings the model rejects rather than letting the request fail.
    ///   - reasoningToggle: a provider-level override for how thinking is
    ///     switched on and off — resellers of this protocol rename the values.
    ///   - streaming: whether the response should stream. `false` produces the
    ///     body for a complete-response request.
    public static func encode(
        _ options: CallOptions,
        model: ModelInfo? = nil,
        reasoningToggle: ProviderInfo.ReasoningToggle? = nil,
        providerId: String = "anthropic",
        streaming: Bool = true
    ) -> EncodedRequest {
        var body: [String: JSONValue] = [
            "model": .string(options.model)
        ]
        if streaming {
            body["stream"] = .bool(true)
        }
        var warnings: [Warning] = []

        // `max_tokens` is required by this API, unlike the others. The fallback
        // is the model's own ceiling where the catalog knows it.
        body["max_tokens"] = .number(Double(
            options.maxOutputTokens ?? model?.maxOutputTokens ?? 4096
        ))

        // System instructions are a top-level field here, not a message.
        let systemText = options.prompt
            .filter { $0.role == .system }
            .map(\.text)
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
        if !systemText.isEmpty {
            body["system"] = .string(systemText)
        }

        body["messages"] = .array(encodeMessages(options.prompt))

        if !options.tools.isEmpty {
            body["tools"] = .array(options.tools.map(encodeTool))
        }
        if let choice = options.toolChoice {
            body["tool_choice"] = encodeToolChoice(choice)
        }
        if !options.stopSequences.isEmpty {
            body["stop_sequences"] = .array(options.stopSequences.map(JSONValue.string))
        }

        // Newer Anthropic models reject sampling parameters outright — a 400,
        // not a silent ignore. Dropping them here turns a hard failure into a
        // warning the caller can act on.
        if let temperature = options.temperature {
            if model?.supportsTemperature == false {
                warnings.append(Warning(
                    message: "\(options.model) does not accept temperature; dropped. Use the effort setting instead.",
                    setting: "temperature"
                ))
            } else {
                body["temperature"] = .number(temperature)
            }
        }
        if let topP = options.topP {
            if model?.supportsTemperature == false {
                warnings.append(Warning(
                    message: "\(options.model) does not accept top_p; dropped.",
                    setting: "topP"
                ))
            } else {
                body["top_p"] = .number(topP)
            }
        }

        if let thinking = options.thinking {
            warnings += applyThinking(
                thinking.plan(model: model, modelId: options.model),
                toggle: reasoningToggle,
                to: &body
            )
        }

        // Provider options merge last so a caller can always override.
        for (key, value) in options.options(for: providerId) {
            body[key] = value
        }

        return EncodedRequest(body: .object(body), warnings: warnings)
    }

    // MARK: - Thinking

    /// Writes the `thinking` block, in whichever of this protocol's three
    /// shapes the model accepts.
    ///
    /// Newer models take `{"type": "adaptive"}` with depth as a separate
    /// `effort`; the 4.x generation takes `{"type": "enabled"}` with a token
    /// budget, which must stay below `max_tokens` because both come out of the
    /// same allowance. Off is `{"type": "disabled"}` — accepted on models that
    /// can be quieted, rejected on the ones that always think, which is why the
    /// plan decides rather than the encoder.
    private static func applyThinking(
        _ plan: ThinkingPlan,
        toggle: ProviderInfo.ReasoningToggle?,
        to body: inout [String: JSONValue]
    ) -> [Warning] {
        guard plan.enabled else {
            guard plan.canDisable else { return plan.warnings }
            if let toggle, let disabled = toggle.disabled {
                toggle.apply(disabled, to: &body)
            } else {
                body["thinking"] = .object(["type": .string("disabled")])
            }
            return plan.warnings
        }

        if let budget = plan.budgetTokens {
            // The budget shares the output allowance with the answer, so a
            // budget at or above `max_tokens` leaves nothing to answer with.
            let cap = body["max_tokens"]?.intValue ?? Int.max
            let bounded = Swift.max(1024, Swift.min(budget, cap - 1024))
            body["thinking"] = .object([
                "type": .string("enabled"),
                "budget_tokens": .number(Double(bounded)),
            ])
        } else if let toggle, let enabled = toggle.enabled {
            toggle.apply(enabled, to: &body)
        } else {
            body["thinking"] = .object(["type": .string("adaptive")])
        }

        if let effort = plan.effort {
            // Effort is nested under `output_config`, not top-level.
            var outputConfig = body["output_config"]?.objectValue ?? [:]
            outputConfig["effort"] = .string(effort)
            body["output_config"] = .object(outputConfig)
        }

        return plan.warnings
    }

    // MARK: - Messages

    private static func encodeMessages(_ prompt: Prompt) -> [JSONValue] {
        var messages: [(role: String, content: [JSONValue], hasToolResults: Bool)] = []

        for message in prompt where message.role != .system {
            // Tool results ride in a `user` turn here rather than a role of
            // their own, which is where the OpenAI shape differs.
            let role = message.role == .tool ? "user" : message.role.rawValue
            var content = message.content.compactMap(encodeContent)

            guard !content.isEmpty else { continue }

            // Anthropic requires tool_result blocks to lead the user content
            // immediately following tool_use. Stable partitioning retains the
            // order among results and among any following user content.
            if role == "user" {
                let results = content.filter { $0["type"]?.stringValue == "tool_result" }
                let remainder = content.filter { $0["type"]?.stringValue != "tool_result" }
                content = results + remainder
            }
            let hasToolResults = content.contains { $0["type"]?.stringValue == "tool_result" }

            // Consecutive tool messages are one Anthropic user turn. A user
            // message immediately following them is appended after the results,
            // preserving the required tool-result-first ordering.
            if role == "user", messages.last?.role == role, messages.last?.hasToolResults == true {
                messages[messages.count - 1].content.append(contentsOf: content)
                messages[messages.count - 1].hasToolResults =
                    messages[messages.count - 1].hasToolResults || hasToolResults
                let all = messages[messages.count - 1].content
                messages[messages.count - 1].content =
                    all.filter { $0["type"]?.stringValue == "tool_result" }
                    + all.filter { $0["type"]?.stringValue != "tool_result" }
            } else {
                messages.append((role, content, hasToolResults))
            }
        }

        return messages.map { message in
            .object(["role": .string(message.role), "content": .array(message.content)])
        }
    }

    private static func encodeContent(_ part: ContentPart) -> JSONValue? {
        switch part {
        case .text(let text):
            guard !text.isEmpty else { return nil }
            return .object(["type": "text", "text": .string(text)])

        case .reasoning(let text, let metadata):
            let anthropic = metadata?["anthropic"]
            if anthropic?["blockType"]?.stringValue == "redacted_thinking"
                || anthropic?["redactedData"] != nil {
                if let wireBlock = anthropic?["wireBlock"] {
                    return wireBlock
                }
                guard let data = anthropic?["redactedData"] else { return nil }
                return .object(["type": "redacted_thinking", "data": data])
            }

            // The signature must be echoed back byte-for-byte; the API rejects
            // a thinking block that has been edited or reconstructed.
            var block = anthropic?["wireBlock"]?.objectValue ?? ["type": "thinking"]
            block["type"] = .string("thinking")
            block["thinking"] = .string(text)
            if let signature = anthropic?["signature"] {
                block["signature"] = signature
            }
            return .object(block)

        case .toolCall(let call):
            return .object([
                "type": "tool_use",
                "id": .string(call.toolCallId),
                "name": .string(call.toolName),
                "input": (try? JSONValue.decode(from: call.input)) ?? .object([:]),
            ])

        case .toolResult(let result):
            var block: [String: JSONValue] = [
                "type": "tool_result",
                "tool_use_id": .string(result.toolCallId),
                "content": result.content.map { .array($0.map(encodeToolResultContent)) }
                    ?? .string(stringify(result.result)),
            ]
            if result.isError { block["is_error"] = .bool(true) }
            return .object(block)

        case .file(let file):
            return encodeFile(file)
        }
    }

    private static func encodeToolResultContent(_ part: ToolResultContent) -> JSONValue {
        switch part {
        case .text(let text):
            .object(["type": "text", "text": .string(text)])
        case .file(let file):
            encodeFile(file)
        }
    }

    private static func encodeFile(_ file: FilePart) -> JSONValue {
        // PDFs use the `document` block; everything else is an image.
        let type = file.mediaType == "application/pdf" ? "document" : "image"

        let source: JSONValue = switch file.data {
        case .base64(let data):
            .object([
                "type": "base64",
                "media_type": .string(file.mediaType),
                "data": .string(data),
            ])
        case .url(let url):
            .object(["type": "url", "url": .string(url)])
        }

        return .object(["type": .string(type), "source": source])
    }

    private static func encodeTool(_ tool: ToolDefinition) -> JSONValue {
        var encoded: [String: JSONValue] = [
            "name": .string(tool.name),
            "input_schema": tool.inputSchema,
        ]
        if let description = tool.description {
            encoded["description"] = .string(description)
        }
        if tool.strict {
            encoded["strict"] = .bool(true)
        }
        return .object(encoded)
    }

    private static func encodeToolChoice(_ choice: ToolChoice) -> JSONValue {
        switch choice {
        case .auto: .object(["type": "auto"])
        case .none: .object(["type": "none"])
        case .required: .object(["type": "any"])
        case .tool(let name): .object(["type": "tool", "name": .string(name)])
        }
    }

    /// Tool results are text on the wire. A string result is passed through
    /// as-is so it does not arrive at the model wrapped in quotes.
    static func stringify(_ value: JSONValue) -> String {
        if case .string(let text) = value { return text }
        return (try? value.encodedString()) ?? ""
    }
}
