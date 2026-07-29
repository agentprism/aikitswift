import Foundation

/// Encodes ``CallOptions`` into a Google Generative AI request body.
///
/// Gemini renames nearly everything: messages are `contents`, the assistant is
/// `model`, sampling settings live under `generationConfig`, and tool results
/// are a part type rather than a role.
public enum GoogleGenerativeAIRequest {

    public static func encode(_ options: CallOptions, model: ModelInfo? = nil) -> EncodedRequest {
        var body: [String: JSONValue] = [:]
        var warnings: [Warning] = []

        // System instructions are a separate top-level field, not a turn.
        let systemText = options.prompt
            .filter { $0.role == .system }
            .map(\.text)
            .filter { !$0.isEmpty }
            .joined(separator: "\n\n")
        if !systemText.isEmpty {
            body["systemInstruction"] = .object([
                "parts": .array([.object(["text": .string(systemText)])])
            ])
        }

        body["contents"] = .array(encodeContents(options.prompt))

        if !options.tools.isEmpty {
            // Every function is declared inside one `tools` entry rather than
            // one entry per function.
            body["tools"] = .array([.object([
                "functionDeclarations": .array(options.tools.map(encodeTool))
            ])])
        }
        if let choice = options.toolChoice {
            body["toolConfig"] = encodeToolChoice(choice)
        }

        var generation: [String: JSONValue] = [:]
        if let maxTokens = options.maxOutputTokens {
            generation["maxOutputTokens"] = .number(Double(maxTokens))
        }
        if let temperature = options.temperature {
            if model?.supportsTemperature == false {
                warnings.append(Warning(
                    message: "\(options.model) does not accept temperature; dropped.",
                    setting: "temperature"
                ))
            } else {
                generation["temperature"] = .number(temperature)
            }
        }
        if let topP = options.topP { generation["topP"] = .number(topP) }
        if let topK = options.topK { generation["topK"] = .number(Double(topK)) }
        if !options.stopSequences.isEmpty {
            generation["stopSequences"] = .array(options.stopSequences.map(JSONValue.string))
        }

        if let thinking = options.thinking {
            let plan = thinking.plan(model: model, modelId: options.model)
            warnings += plan.warnings

            // Gemini splits by generation: 3.x takes a named level, 2.5 takes a
            // token budget where zero means off. `thinkingConfig` is nested
            // inside `generationConfig`, not top-level.
            var thinkingConfig: [String: JSONValue] = [:]

            if plan.enabled {
                if let effort = plan.effort {
                    thinkingConfig["thinkingLevel"] = .string(effort)
                } else if let budget = plan.budgetTokens {
                    thinkingConfig["thinkingBudget"] = .number(Double(budget))
                } else {
                    // Let the model size its own budget.
                    thinkingConfig["thinkingBudget"] = .number(-1)
                }
            } else if plan.canDisable {
                if let off = plan.offEffort {
                    thinkingConfig["thinkingLevel"] = .string(off)
                } else {
                    thinkingConfig["thinkingBudget"] = .number(0)
                }
            }

            if !thinkingConfig.isEmpty {
                generation["thinkingConfig"] = .object(thinkingConfig)
            }
        }

        if !generation.isEmpty {
            body["generationConfig"] = .object(generation)
        }

        for (key, value) in options.options(for: "google") {
            body[key] = value
        }

        return EncodedRequest(body: .object(body), warnings: warnings)
    }

    // MARK: - Contents

    private static func encodeContents(_ prompt: Prompt) -> [JSONValue] {
        var contents: [JSONValue] = []

        for message in prompt where message.role != .system {
            // Gemini knows only `user` and `model`; tool results are `user`
            // parts rather than a role of their own.
            let role = message.role == .assistant ? "model" : "user"
            let parts = message.content.compactMap(encodePart)

            guard !parts.isEmpty else { continue }
            contents.append(.object(["role": .string(role), "parts": .array(parts)]))
        }

        return contents
    }

    private static func encodePart(_ part: ContentPart) -> JSONValue? {
        switch part {
        case .text(let text):
            guard !text.isEmpty else { return nil }
            return .object(["text": .string(text)])

        case .reasoning(let text, let metadata):
            var encoded: [String: JSONValue] = ["text": .string(text), "thought": .bool(true)]
            // Echoed back unchanged so multi-turn reasoning survives.
            if let signature = metadata?["google"]?["thoughtSignature"] {
                encoded["thoughtSignature"] = signature
            }
            return .object(encoded)

        case .toolCall(let call):
            return .object(["functionCall": .object([
                "name": .string(call.toolName),
                // Arguments are a real object here, not a JSON string.
                "args": (try? JSONValue.decode(from: call.input)) ?? .object([:]),
            ])])

        case .toolResult(let result):
            // Matched by tool *name*, not by call id — Gemini issues no ids.
            return .object(["functionResponse": .object([
                "name": .string(result.toolName),
                "response": wrapResponse(result.result),
            ])])

        case .file(let file):
            switch file.data {
            case .base64(let data):
                return .object(["inlineData": .object([
                    "mimeType": .string(file.mediaType),
                    "data": .string(data),
                ])])
            case .url(let url):
                return .object(["fileData": .object([
                    "mimeType": .string(file.mediaType),
                    "fileUri": .string(url),
                ])])
            }
        }
    }

    /// `functionResponse.response` must be an object. Scalars and arrays are
    /// wrapped rather than sent bare, which the API rejects.
    private static func wrapResponse(_ value: JSONValue) -> JSONValue {
        if case .object = value { return value }
        return .object(["result": value])
    }

    private static func encodeTool(_ tool: ToolDefinition) -> JSONValue {
        var declaration: [String: JSONValue] = [
            "name": .string(tool.name),
            "parameters": tool.inputSchema,
        ]
        if let description = tool.description {
            declaration["description"] = .string(description)
        }
        return .object(declaration)
    }

    private static func encodeToolChoice(_ choice: ToolChoice) -> JSONValue {
        let config: [String: JSONValue] = switch choice {
        case .auto: ["mode": "AUTO"]
        case .none: ["mode": "NONE"]
        case .required: ["mode": "ANY"]
        case .tool(let name): ["mode": "ANY", "allowedFunctionNames": .array([.string(name)])]
        }
        return .object(["functionCallingConfig": .object(config)])
    }
}
