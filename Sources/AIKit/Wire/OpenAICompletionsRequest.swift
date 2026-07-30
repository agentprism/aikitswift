import Foundation

/// Encodes ``CallOptions`` into an OpenAI Chat Completions request body.
///
/// Also serves the thirty-odd providers that imitate this API. Where they
/// diverge it is usually in which optional fields they tolerate, so anything
/// beyond the common core belongs in `providerOptions` rather than here.
public enum OpenAICompletionsRequest {

    /// - Parameter dialect: how this provider deviates from the reference
    ///   API. Defaults to the conservative baseline; pass
    ///   ``CompletionsDialect/forProvider(_:)`` for a catalog provider.
    /// - Parameter reasoningToggle: a provider-level override for how thinking
    ///   is switched on and off, from the catalog.
    /// - Parameter streaming: whether the response should stream. `false`
    ///   produces the body for a complete-response request — and must also omit
    ///   `stream_options`, which providers reject on non-streaming requests.
    public static func encode(
        _ options: CallOptions,
        model: ModelInfo? = nil,
        dialect: CompletionsDialect = .default,
        reasoningToggle: ProviderInfo.ReasoningToggle? = nil,
        streaming: Bool = true
    ) -> EncodedRequest {
        var body: [String: JSONValue] = [
            "model": .string(options.model)
        ]
        var warnings: [Warning] = []

        if streaming {
            body["stream"] = .bool(true)

            // Without this, a provider that supports it returns **no usage at
            // all** on a streamed response — while a provider that does not
            // support it rejects the request outright. The only flag that is
            // dangerous in both directions, which is why it is dialect-driven
            // rather than assumed.
            if dialect.supportsUsageInStreaming {
                body["stream_options"] = .object(["include_usage": .bool(true)])
            }
        }

        body["messages"] = .array(encodeMessages(options.prompt, dialect: dialect))

        if let maxTokens = options.maxOutputTokens {
            // Reasoning models rejected the older `max_tokens` in favour of
            // `max_completion_tokens`, and the budget covers reasoning too.
            let field = (model?.supportsReasoning ?? false)
                ? CompletionsDialect.MaxTokensField.maxCompletionTokens
                : dialect.maxTokensField
            body[field.rawValue] = .number(Double(maxTokens))
        }

        // Rendered into whichever shape this vendor expects — there are at
        // least seven incompatible ones in the catalog, and each has both an
        // on and an off spelling.
        if let thinking = options.thinking {
            warnings += dialect.applyThinking(
                thinking.plan(model: model, modelId: options.model),
                toggle: reasoningToggle,
                to: &body
            )
        }

        if !options.tools.isEmpty {
            body["tools"] = .array(options.tools.map { encodeTool($0, dialect: dialect) })
        }
        if let choice = options.toolChoice {
            body["tool_choice"] = encodeToolChoice(choice)
        }
        if !options.stopSequences.isEmpty {
            body["stop"] = .array(options.stopSequences.map(JSONValue.string))
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
        if options.topK != nil {
            // No equivalent field exists in this API.
            warnings.append(Warning(
                message: "Chat Completions has no top_k parameter; dropped.",
                setting: "topK"
            ))
        }
        if let format = options.responseFormat {
            body["response_format"] = format
        }

        for (key, value) in options.options(for: "openai") {
            body[key] = value
        }

        return EncodedRequest(body: .object(body), warnings: warnings)
    }

    // MARK: - Messages

    private static func encodeMessages(_ prompt: Prompt, dialect: CompletionsDialect) -> [JSONValue] {
        var messages: [JSONValue] = []

        for message in prompt {
            switch message.role {
            case .tool:
                // Each tool result is its own message here, keyed by call id —
                // unlike Anthropic, which batches them into one user turn.
                for part in message.content {
                    guard case .toolResult(let result) = part else { continue }
                    var encoded: [String: JSONValue] = [
                        "role": "tool",
                        "tool_call_id": .string(result.toolCallId),
                        "content": .string(AnthropicMessagesRequest.stringify(result.result)),
                    ]
                    // A few providers reject a tool result that omits the name,
                    // even though the reference API ignores it.
                    if dialect.requiresToolResultName {
                        encoded["name"] = .string(result.toolName)
                    }
                    messages.append(.object(encoded))
                }

            case .assistant:
                messages.append(encodeAssistant(message))

            case .system, .user:
                messages.append(encodeSimple(message))
            }
        }

        return messages
    }

    private static func encodeSimple(_ message: Message) -> JSONValue {
        // A text-only message may use the string form, which every
        // compatible server accepts. The array form is newer and some
        // implementations still reject it.
        let hasAttachments = message.content.contains { if case .file = $0 { return true } else { return false } }

        guard hasAttachments else {
            return .object(["role": .string(message.role.rawValue), "content": .string(message.text)])
        }

        let parts: [JSONValue] = message.content.compactMap { part in
            switch part {
            case .text(let text):
                .object(["type": "text", "text": .string(text)])
            case .file(let file):
                encodeFile(file)
            default:
                nil
            }
        }

        return .object(["role": .string(message.role.rawValue), "content": .array(parts)])
    }

    private static func encodeAssistant(_ message: Message) -> JSONValue {
        var encoded: [String: JSONValue] = ["role": "assistant"]

        let text = message.text
        if !text.isEmpty {
            encoded["content"] = .string(text)
        }

        let calls: [JSONValue] = message.content.compactMap { part in
            guard case .toolCall(let call) = part else { return nil }
            return .object([
                "id": .string(call.toolCallId),
                "type": "function",
                "function": .object([
                    "name": .string(call.toolName),
                    "arguments": .string(call.input),
                ]),
            ])
        }
        if !calls.isEmpty {
            encoded["tool_calls"] = .array(calls)
        }

        // An assistant turn must carry something; a tool-call-only turn still
        // needs the key present for stricter compatible servers.
        if encoded["content"] == nil && calls.isEmpty {
            encoded["content"] = .string("")
        }

        return .object(encoded)
    }

    private static func encodeFile(_ file: FilePart) -> JSONValue {
        let url: String = switch file.data {
        case .url(let value): value
        // Binary content is inlined as a data URL, which is the only form this
        // API accepts for uploaded bytes.
        case .base64(let data): "data:\(file.mediaType);base64,\(data)"
        }

        return .object(["type": "image_url", "image_url": .object(["url": .string(url)])])
    }

    private static func encodeTool(_ tool: ToolDefinition, dialect: CompletionsDialect) -> JSONValue {
        var function: [String: JSONValue] = [
            "name": .string(tool.name),
            "parameters": tool.inputSchema,
        ]
        if let description = tool.description {
            function["description"] = .string(description)
        }
        // Sending `strict` to a provider that does not implement it is a 400,
        // so it is dropped rather than risking the whole request.
        if tool.strict && dialect.supportsStrict {
            function["strict"] = .bool(true)
        }
        return .object(["type": "function", "function": .object(function)])
    }

    private static func encodeToolChoice(_ choice: ToolChoice) -> JSONValue {
        switch choice {
        case .auto: .string("auto")
        case .none: .string("none")
        case .required: .string("required")
        case .tool(let name):
            .object(["type": "function", "function": .object(["name": .string(name)])])
        }
    }
}
