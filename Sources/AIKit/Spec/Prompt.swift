import Foundation

/// Who produced a message.
public enum MessageRole: String, Sendable, Hashable, Codable {
    case system
    case user
    case assistant
    /// Results of tools the client executed, handed back to the model.
    case tool
}

/// Binary or remote content attached to a message.
public struct FilePart: Sendable, Hashable {
    public enum Payload: Sendable, Hashable {
        /// Base64-encoded bytes.
        case base64(String)
        /// A URL the provider fetches itself. Not every provider supports this.
        case url(String)
    }

    public var mediaType: String
    public var data: Payload
    public var filename: String?

    public init(mediaType: String, data: Payload, filename: String? = nil) {
        self.mediaType = mediaType
        self.data = data
        self.filename = filename
    }
}

/// One piece of a message.
///
/// Messages are always lists of parts, even when they hold a single string, so
/// that multimodal content and tool traffic share one shape.
public enum ContentPart: Sendable, Hashable {
    case text(String)
    case file(FilePart)

    /// Model reasoning being replayed on a later turn.
    ///
    /// Carries `providerMetadata` because some providers require an opaque
    /// signature to be echoed back **unchanged** — Anthropic rejects the
    /// request if a thinking block is edited. Reconstructing this part by hand
    /// is how multi-turn reasoning breaks.
    case reasoning(String, providerMetadata: ProviderMetadata? = nil)

    /// A tool call the model made on a previous turn.
    case toolCall(ToolCall)

    /// The result of executing a tool, being returned to the model.
    case toolResult(ToolResult)
}

/// One message in a conversation.
public struct Message: Sendable, Hashable {
    public var role: MessageRole
    public var content: [ContentPart]
    /// Provider-specific options for this message, namespaced by provider id.
    ///
    /// Cache breakpoints live here, for instance.
    public var providerOptions: ProviderMetadata?

    public init(role: MessageRole, content: [ContentPart], providerOptions: ProviderMetadata? = nil) {
        self.role = role
        self.content = content
        self.providerOptions = providerOptions
    }
}

// MARK: - Conveniences

extension Message {
    public static func system(_ text: String) -> Message {
        Message(role: .system, content: [.text(text)])
    }

    public static func user(_ text: String) -> Message {
        Message(role: .user, content: [.text(text)])
    }

    public static func assistant(_ text: String) -> Message {
        Message(role: .assistant, content: [.text(text)])
    }

    /// A tool result being returned to the model.
    public static func toolResult(
        toolCallId: String,
        toolName: String,
        result: JSONValue,
        isError: Bool = false
    ) -> Message {
        Message(role: .tool, content: [.toolResult(ToolResult(
            toolCallId: toolCallId,
            toolName: toolName,
            result: result,
            isError: isError
        ))])
    }

    /// A structured, potentially multimodal tool result.
    public static func toolResult(
        toolCallId: String,
        toolName: String,
        content: [ToolResultContent],
        isError: Bool = false
    ) -> Message {
        Message(role: .tool, content: [.toolResult(ToolResult(
            toolCallId: toolCallId,
            toolName: toolName,
            result: .null,
            content: content,
            isError: isError
        ))])
    }

    /// All text in this message, concatenated.
    public var text: String {
        content.compactMap { if case .text(let value) = $0 { return value } else { return nil } }
            .joined()
    }
}

/// A conversation: an ordered list of messages.
public typealias Prompt = [Message]
