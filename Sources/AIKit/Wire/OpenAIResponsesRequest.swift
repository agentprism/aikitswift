import Foundation

/// Encodes ``CallOptions`` into an OpenAI Responses request body.
///
/// Flatter than Chat Completions: `input` is a list of items rather than
/// messages with nested content, and tool calls and their results are items in
/// that same list rather than fields on a message.
public enum OpenAIResponsesRequest {

    /// - Parameter streaming: whether the response should stream. `false`
    ///   produces the body for a complete-response request.
    public static func encode(
        _ options: CallOptions,
        model: ModelInfo? = nil,
        streaming: Bool = true
    ) -> EncodedRequest {
        var body: [String: JSONValue] = [
            "model": .string(options.model)
        ]
        if streaming {
            body["stream"] = .bool(true)
        }
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
            body["tools"] = .array(encodeTools(options.tools))
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

        if let thinking = options.thinking {
            // One nested field carries both directions here: an effort of
            // `none` is how a reasoning model is told not to reason. Models
            // whose vocabulary lacks it cannot be quieted, and say so.
            let plan = thinking.plan(model: model, modelId: options.model)
            warnings += plan.warnings

            if plan.enabled {
                if let effort = plan.effort {
                    body["reasoning"] = .object(["effort": .string(effort)])
                }
            } else if plan.canDisable, let off = plan.offEffort {
                body["reasoning"] = .object(["effort": .string(off)])
            }
        }

        for (key, value) in options.options(for: "openai") {
            body[key] = value
        }

        return EncodedRequest(body: .object(body), warnings: warnings)
    }

    // MARK: - Input items

    private static func encodeInput(_ prompt: Prompt) -> [JSONValue] {
        var items: [JSONValue] = []
        var outputTypeByCallId: [String: String] = [:]

        for message in prompt where message.role != .system {
            switch message.role {
            case .tool:
                for part in message.content {
                    guard case .toolResult(let result) = part else { continue }
                    let callType = result.providerMetadata?["openai"]?["callType"]?.stringValue
                        ?? outputTypeByCallId[result.toolCallId]
                    var item: [String: JSONValue] = [
                        "type": .string(
                            callType == "custom_tool_call"
                                ? "custom_tool_call_output"
                                : "function_call_output"
                        ),
                        // `call_id`, not the streaming item id.
                        "call_id": .string(result.toolCallId),
                        "output": encodeToolOutput(result),
                    ]
                    if let itemId = result.providerMetadata?["openai"]?["itemId"] {
                        item["id"] = itemId
                    }
                    items.append(.object(item))
                }

            case .assistant:
                // Responses output items are already valid next-turn input.
                // Replaying the provider-owned shapes is the only lossless way
                // to retain encrypted reasoning, item identity, status, phase,
                // namespaces, and fields added by newer protocol revisions.
                if let replay = message.providerOptions?["openai"]?["outputItems"]?.arrayValue {
                    for item in replay {
                        guard let callId = item["call_id"]?.stringValue,
                              let type = item["type"]?.stringValue,
                              type == "function_call" || type == "custom_tool_call" else { continue }
                        outputTypeByCallId[callId] = type
                    }
                    items.append(contentsOf: replay)
                    continue
                }

                // Tool calls are peers of the message, not fields on it.
                for part in message.content {
                    guard case .toolCall(let call) = part else { continue }
                    if let item = call.providerMetadata?["openai"]?["item"] {
                        if let callId = item["call_id"]?.stringValue,
                           let type = item["type"]?.stringValue {
                            outputTypeByCallId[callId] = type
                        }
                        items.append(item)
                        continue
                    }

                    var item: [String: JSONValue] = [
                        "type": "function_call",
                        "call_id": .string(call.toolCallId),
                        "name": .string(call.toolName),
                        "arguments": .string(call.input),
                    ]
                    if let namespace = call.namespace {
                        item["namespace"] = .string(namespace)
                    }
                    outputTypeByCallId[call.toolCallId] = "function_call"
                    items.append(.object(item))
                }

                // A hand-built reasoning part can still carry a complete
                // provider item even when the message did not come directly
                // from `AIResponse.assistantMessage`.
                for part in message.content {
                    guard case .reasoning(_, let metadata) = part,
                          let item = metadata?["openai"]?["item"] else { continue }
                    items.append(item)
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

    private static func encodeTools(_ tools: [ToolDefinition]) -> [JSONValue] {
        var encoded: [JSONValue] = []
        var namespaceIndices: [String: Int] = [:]

        for tool in tools {
            guard let namespace = tool.namespace else {
                encoded.append(encodeFunction(tool))
                continue
            }

            if let index = namespaceIndices[namespace.name] {
                var group = encoded[index].objectValue ?? [:]
                var functions = group["tools"]?.arrayValue ?? []
                functions.append(encodeFunction(tool))
                group["tools"] = .array(functions)
                encoded[index] = .object(group)
            } else {
                var group: [String: JSONValue] = [
                    "type": "namespace",
                    "name": .string(namespace.name),
                    "tools": .array([encodeFunction(tool)]),
                ]
                group["description"] = .string(namespace.description)
                namespaceIndices[namespace.name] = encoded.count
                encoded.append(.object(group))
            }
        }

        return encoded
    }

    private static func encodeFunction(_ tool: ToolDefinition) -> JSONValue {
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

    private static func encodeToolOutput(_ result: ToolResult) -> JSONValue {
        if let output = result.providerMetadata?["openai"]?["output"] {
            return output
        }
        guard let content = result.content else {
            return .string(AnthropicMessagesRequest.stringify(result.result))
        }
        return .array(content.map { part in
            switch part {
            case .text(let text):
                return .object(["type": "input_text", "text": .string(text)])
            case .file(let file):
                return encodeFile(file)
            }
        })
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
