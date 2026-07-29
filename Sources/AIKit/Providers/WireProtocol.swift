import Foundation

/// A streaming wire format.
///
/// The catalog holds ~50 providers that resolve to these few values. That ratio
/// is the entire architectural argument: implement a protocol once and every
/// provider pointing at it works.
public enum WireProtocol: String, Sendable, Hashable, Codable, CaseIterable {
    /// Anthropic Messages API.
    case anthropicMessages = "anthropic"
    /// OpenAI Chat Completions, and the many endpoints that imitate it.
    case openAICompletions = "openai"
    /// OpenAI Responses API.
    case openAIResponses = "openai-responses"
    /// OpenAI Codex, which speaks Responses with a different auth flow.
    case openAICodex = "openai-codex"
    /// Google Generative AI (Gemini).
    case googleGenerativeAI = "gemini"

    /// A mapper for this protocol, ready to decode one response.
    public func makeMapper() -> AnyWireMapper {
        switch self {
        case .anthropicMessages: .anthropic(AnthropicMessagesWire())
        case .openAICompletions: .openAICompletions(OpenAICompletionsWire())
        // Codex differs from Responses in how a request is authenticated, not
        // in how a response is shaped, so the same mapper serves both.
        case .openAIResponses, .openAICodex: .openAIResponses(OpenAIResponsesWire())
        case .googleGenerativeAI: .google(GoogleGenerativeAIWire())
        }
    }
}

/// A type-erased ``WireMapper``.
///
/// An enum rather than an existential because `WireMapper` has mutating
/// requirements, and boxing those costs either a class allocation per stream or
/// a great deal of ceremony. The set of protocols is closed and small, so an
/// enum is both cheaper and clearer.
public enum AnyWireMapper: WireMapper {
    case anthropic(AnthropicMessagesWire)
    case openAICompletions(OpenAICompletionsWire)
    case openAIResponses(OpenAIResponsesWire)
    case google(GoogleGenerativeAIWire)

    /// Defaults to Chat Completions, the most widely spoken protocol.
    public init() { self = .openAICompletions(OpenAICompletionsWire()) }

    public mutating func map(chunk: JSONValue) -> [StreamPart] {
        switch self {
        case .anthropic(var wire):
            defer { self = .anthropic(wire) }
            return wire.map(chunk: chunk)
        case .openAICompletions(var wire):
            defer { self = .openAICompletions(wire) }
            return wire.map(chunk: chunk)
        case .openAIResponses(var wire):
            defer { self = .openAIResponses(wire) }
            return wire.map(chunk: chunk)
        case .google(var wire):
            defer { self = .google(wire) }
            return wire.map(chunk: chunk)
        }
    }

    public mutating func map(rawJSON: String) -> [StreamPart] {
        switch self {
        case .anthropic(var wire):
            defer { self = .anthropic(wire) }
            return wire.map(rawJSON: rawJSON)
        case .openAICompletions(var wire):
            defer { self = .openAICompletions(wire) }
            return wire.map(rawJSON: rawJSON)
        case .openAIResponses(var wire):
            defer { self = .openAIResponses(wire) }
            return wire.map(rawJSON: rawJSON)
        case .google(var wire):
            defer { self = .google(wire) }
            return wire.map(rawJSON: rawJSON)
        }
    }

    public mutating func finish() -> [StreamPart] {
        switch self {
        case .anthropic(var wire):
            defer { self = .anthropic(wire) }
            return wire.finish()
        case .openAICompletions(var wire):
            defer { self = .openAICompletions(wire) }
            return wire.finish()
        case .openAIResponses(var wire):
            defer { self = .openAIResponses(wire) }
            return wire.finish()
        case .google(var wire):
            defer { self = .google(wire) }
            return wire.finish()
        }
    }
}
