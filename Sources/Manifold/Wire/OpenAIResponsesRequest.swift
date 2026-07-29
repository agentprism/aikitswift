import Foundation

/// Encodes ``CallOptions`` into an OpenAI Responses request body.
///
/// Flatter than Chat Completions: `input` is a list of items rather than
/// messages with nested content, and tool calls and their results are items in
/// that same list rather than fields on a message.
public enum OpenAIResponsesRequest {

    public static func encode(_ options: CallOptions, model: ModelInfo? = nil) -> EncodedRequest {
        var body: [String: JSONValue] = [
            "model": .string(options.model),
            "stream": .bool(true),
        ]
        var warnings: [Warning] = []

        // System instructions are a top-level string, not an input item.
        let systemText = options.prompt
            .filter { $0.role == .system }
            .map(\.text)
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
        if !systemText.isEmpty {
            body["instructions"] = .string(systemText)
        }

        body["input"] = .array(encodeInput(options.prompt))

        if let maxTokens = options.maxOutputTokens {
            body["max_output_tokens"] = .number(Double(maxTokens))
        }
        if !options.tools.isEmpty {
            body["tools"] = .array(options.tools.map(encodeTool))
        }
        if let choice = options.toolChoice {
            body["tool_choice"] = encodeToolChoice(choice)
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
            // The Responses API dropped stop sequences.
            warnings.append(Warning(
                message: "The Responses API has no stop-sequence parameter; dropped.",
                setting: "stopSequences"
            ))
        }

        for (key, value) in options.options(for: "openai") {
            body[key] = value
        }

        return EncodedRequest(body: .object(body), warnings: warnings)
    }

    // MARK: - Input items

    private static func encodeInput(_ prompt: Prompt) -> [JSONValue] {
        var items: [JSONValue] = []

        for message in prompt where message.role != .system {
            switch message.role {
            case .tool:
                for part in message.content {
                    guard case .toolResult(let result) = part else { continue }
                    items.append(.object([
                        "type": "function_call_output",
                        // `call_id`, not the streaming item id.
                        "call_id": .string(result.toolCallId),
                        "output": .string(AnthropicMessagesRequest.stringify(result.result)),
                    ]))
                }

            case .assistant:
                // Tool calls are peers of the message, not fields on it.
                for part in message.content {
                    guard case .toolCall(let call) = part else { continue }
                    items.append(.object([
                        "type": "function_call",
                        "call_id": .string(call.toolCallId),
                        "name": .string(call.toolName),
                        "arguments": .string(call.input),
                    ]))
                }

                let text = message.text
                if !text.isEmpty {
                    items.append(.object([
                        "role": "assistant",
                        "content": .array([.object(["type": "output_text", "text": .string(text)])]),
                    ]))
                }

            case .user, .system:
                let parts: [JSONValue] = message.content.compactMap { part in
                    switch part {
                    case .text(let text):
                        // Input and output text are different part types here.
                        .object(["type": "input_text", "text": .string(text)])
                    case .file(let file):
                        encodeFile(file)
                    default:
                        nil
                    }
                }
                guard !parts.isEmpty else { continue }
                items.append(.object(["role": "user", "content": .array(parts)]))
            }
        }

        return items
    }

    private static func encodeFile(_ file: FilePart) -> JSONValue {
        let url: String = switch file.data {
        case .url(let value): value
        case .base64(let data): "data:\(file.mediaType);base64,\(data)"
        }

        if file.mediaType == "application/pdf" {
            var encoded: [String: JSONValue] = ["type": "input_file", "file_data": .string(url)]
            if let filename = file.filename {
                encoded["filename"] = .string(filename)
            }
            return .object(encoded)
        }

        return .object(["type": "input_image", "image_url": .string(url)])
    }

    private static func encodeTool(_ tool: ToolDefinition) -> JSONValue {
        // Flat, unlike Completions where the definition nests under `function`.
        var encoded: [String: JSONValue] = [
            "type": "function",
            "name": .string(tool.name),
            "parameters": tool.inputSchema,
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
        case .auto: .string("auto")
        case .none: .string("none")
        case .required: .string("required")
        case .tool(let name): .object(["type": "function", "name": .string(name)])
        }
    }
}
