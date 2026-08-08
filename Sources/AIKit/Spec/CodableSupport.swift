import Foundation

extension FilePart.Payload: Codable {
    private enum CodingKeys: String, CodingKey {
        case kind
        case value
    }

    private enum Kind: String, Codable {
        case base64
        case url
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(Kind.self, forKey: .kind) {
        case .base64:
            self = .base64(try container.decode(String.self, forKey: .value))
        case .url:
            self = .url(try container.decode(String.self, forKey: .value))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .base64(let value):
            try container.encode(Kind.base64, forKey: .kind)
            try container.encode(value, forKey: .value)
        case .url(let value):
            try container.encode(Kind.url, forKey: .kind)
            try container.encode(value, forKey: .value)
        }
    }
}

extension FilePart: Codable {
    private enum CodingKeys: String, CodingKey {
        case mediaType
        case data
        case filename
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            mediaType: try container.decode(String.self, forKey: .mediaType),
            data: try container.decode(Payload.self, forKey: .data),
            filename: try container.decodeIfPresent(String.self, forKey: .filename)
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(mediaType, forKey: .mediaType)
        try container.encode(data, forKey: .data)
        try container.encodeIfPresent(filename, forKey: .filename)
    }
}

extension ContentPart: Codable {
    private enum CodingKeys: String, CodingKey {
        case kind
        case text
        case file
        case providerMetadata
        case toolCall
        case toolResult
    }

    private enum Kind: String, Codable {
        case text
        case file
        case reasoning
        case toolCall
        case toolResult
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        switch try container.decode(Kind.self, forKey: .kind) {
        case .text:
            self = .text(try container.decode(String.self, forKey: .text))
        case .file:
            self = .file(try container.decode(FilePart.self, forKey: .file))
        case .reasoning:
            self = .reasoning(
                try container.decode(String.self, forKey: .text),
                providerMetadata: try container.decodeIfPresent(ProviderMetadata.self, forKey: .providerMetadata)
            )
        case .toolCall:
            self = .toolCall(try container.decode(ToolCall.self, forKey: .toolCall))
        case .toolResult:
            self = .toolResult(try container.decode(ToolResult.self, forKey: .toolResult))
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .text(let value):
            try container.encode(Kind.text, forKey: .kind)
            try container.encode(value, forKey: .text)
        case .file(let value):
            try container.encode(Kind.file, forKey: .kind)
            try container.encode(value, forKey: .file)
        case .reasoning(let value, let providerMetadata):
            try container.encode(Kind.reasoning, forKey: .kind)
            try container.encode(value, forKey: .text)
            try container.encodeIfPresent(providerMetadata, forKey: .providerMetadata)
        case .toolCall(let value):
            try container.encode(Kind.toolCall, forKey: .kind)
            try container.encode(value, forKey: .toolCall)
        case .toolResult(let value):
            try container.encode(Kind.toolResult, forKey: .kind)
            try container.encode(value, forKey: .toolResult)
        }
    }
}

extension Message: Codable {
    private enum CodingKeys: String, CodingKey {
        case role
        case content
        case providerOptions
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            role: try container.decode(MessageRole.self, forKey: .role),
            content: try container.decode([ContentPart].self, forKey: .content),
            providerOptions: try container.decodeIfPresent(ProviderMetadata.self, forKey: .providerOptions)
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        try container.encode(content, forKey: .content)
        try container.encodeIfPresent(providerOptions, forKey: .providerOptions)
    }
}

extension ToolCall: Codable {
    private enum CodingKeys: String, CodingKey {
        case toolCallId
        case toolName
        case input
        case providerExecuted
        case dynamic
        case providerMetadata
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            toolCallId: try container.decode(String.self, forKey: .toolCallId),
            toolName: try container.decode(String.self, forKey: .toolName),
            input: try container.decode(String.self, forKey: .input),
            providerExecuted: try container.decodeIfPresent(Bool.self, forKey: .providerExecuted) ?? false,
            dynamic: try container.decodeIfPresent(Bool.self, forKey: .dynamic) ?? false,
            providerMetadata: try container.decodeIfPresent(ProviderMetadata.self, forKey: .providerMetadata)
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(toolCallId, forKey: .toolCallId)
        try container.encode(toolName, forKey: .toolName)
        try container.encode(input, forKey: .input)
        try container.encode(providerExecuted, forKey: .providerExecuted)
        try container.encode(dynamic, forKey: .dynamic)
        try container.encodeIfPresent(providerMetadata, forKey: .providerMetadata)
    }
}

extension ToolResult: Codable {
    private enum CodingKeys: String, CodingKey {
        case toolCallId
        case toolName
        case result
        case isError
        case preliminary
        case dynamic
        case providerMetadata
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            toolCallId: try container.decode(String.self, forKey: .toolCallId),
            toolName: try container.decode(String.self, forKey: .toolName),
            result: try container.decode(JSONValue.self, forKey: .result),
            isError: try container.decodeIfPresent(Bool.self, forKey: .isError) ?? false,
            preliminary: try container.decodeIfPresent(Bool.self, forKey: .preliminary) ?? false,
            dynamic: try container.decodeIfPresent(Bool.self, forKey: .dynamic) ?? false,
            providerMetadata: try container.decodeIfPresent(ProviderMetadata.self, forKey: .providerMetadata)
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(toolCallId, forKey: .toolCallId)
        try container.encode(toolName, forKey: .toolName)
        try container.encode(result, forKey: .result)
        try container.encode(isError, forKey: .isError)
        try container.encode(preliminary, forKey: .preliminary)
        try container.encode(dynamic, forKey: .dynamic)
        try container.encodeIfPresent(providerMetadata, forKey: .providerMetadata)
    }
}

extension FinishReason: Codable {
    private enum CodingKeys: String, CodingKey {
        case unified
        case raw
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            unified: try container.decode(Unified.self, forKey: .unified),
            raw: try container.decodeIfPresent(String.self, forKey: .raw)
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(unified, forKey: .unified)
        try container.encodeIfPresent(raw, forKey: .raw)
    }
}

extension Usage.Input: Codable {
    private enum CodingKeys: String, CodingKey {
        case total
        case noCache
        case cacheRead
        case cacheWrite
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            total: try container.decodeIfPresent(Int.self, forKey: .total),
            noCache: try container.decodeIfPresent(Int.self, forKey: .noCache),
            cacheRead: try container.decodeIfPresent(Int.self, forKey: .cacheRead),
            cacheWrite: try container.decodeIfPresent(Int.self, forKey: .cacheWrite)
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(total, forKey: .total)
        try container.encodeIfPresent(noCache, forKey: .noCache)
        try container.encodeIfPresent(cacheRead, forKey: .cacheRead)
        try container.encodeIfPresent(cacheWrite, forKey: .cacheWrite)
    }
}

extension Usage.Output: Codable {
    private enum CodingKeys: String, CodingKey {
        case total
        case text
        case reasoning
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            total: try container.decodeIfPresent(Int.self, forKey: .total),
            text: try container.decodeIfPresent(Int.self, forKey: .text),
            reasoning: try container.decodeIfPresent(Int.self, forKey: .reasoning)
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encodeIfPresent(total, forKey: .total)
        try container.encodeIfPresent(text, forKey: .text)
        try container.encodeIfPresent(reasoning, forKey: .reasoning)
    }
}

extension Usage: Codable {
    private enum CodingKeys: String, CodingKey {
        case inputTokens
        case outputTokens
        case raw
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.init(
            inputTokens: try container.decode(Input.self, forKey: .inputTokens),
            outputTokens: try container.decode(Output.self, forKey: .outputTokens),
            raw: try container.decodeIfPresent(JSONValue.self, forKey: .raw)
        )
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(inputTokens, forKey: .inputTokens)
        try container.encode(outputTokens, forKey: .outputTokens)
        try container.encodeIfPresent(raw, forKey: .raw)
    }
}
