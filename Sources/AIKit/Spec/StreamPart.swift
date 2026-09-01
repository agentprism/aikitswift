import Foundation

/// Provider-specific metadata, namespaced by provider id.
///
/// Anything a provider emits that has no home in the normalized shape lands
/// here under its own key (`["anthropic": ["signature": ...]]`), so a caller
/// that knows the provider can still reach it without the spec growing a
/// field for every vendor.
public typealias ProviderMetadata = [String: [String: JSONValue]]

/// A non-fatal problem with the request — an unsupported setting that was
/// dropped, a parameter the provider silently ignored.
public struct Warning: Sendable, Hashable {
    public var message: String
    public var setting: String?

    public init(message: String, setting: String? = nil) {
        self.message = message
        self.setting = setting
    }
}

/// Metadata about the response itself, delivered as soon as it is known
/// rather than held back until the stream finishes.
public struct ResponseMetadata: Sendable, Hashable {
    public var id: String?
    public var timestamp: Date?
    /// The model that actually served the response, which is not always the
    /// model that was requested — server-side fallbacks can reroute a turn.
    public var modelId: String?
    /// The requested provider/API/model identity for this response.
    ///
    /// ``AIClient`` attaches this at the normalized response boundary.
    /// `modelId` remains the actual model reported by the server.
    public var producer: ModelDestination?

    public init(
        id: String? = nil,
        timestamp: Date? = nil,
        modelId: String? = nil,
        producer: ModelDestination? = nil
    ) {
        self.id = id
        self.timestamp = timestamp
        self.modelId = modelId
        self.producer = producer
    }
}

/// A complete tool call, emitted once its input has finished streaming.
public struct ToolCall: Sendable, Hashable {
    public var toolCallId: String
    public var toolName: String
    /// The Responses function namespace, when the provider grouped this
    /// function under one. The namespace is distinct from the function name.
    public var namespace: String?
    /// The arguments as a JSON *string*, not a parsed object.
    ///
    /// It stays a string because that is what streamed in, character by
    /// character, and re-encoding a parsed value would not reproduce the
    /// original bytes. Parse it at the point of use.
    public var input: String
    /// `true` when the provider ran the tool server-side and the client has
    /// nothing to execute.
    public var providerExecuted: Bool
    /// `true` for tools defined at runtime rather than declared up front
    /// (MCP tools, for instance).
    public var dynamic: Bool
    public var providerMetadata: ProviderMetadata?

    public init(
        toolCallId: String,
        toolName: String,
        namespace: String? = nil,
        input: String,
        providerExecuted: Bool = false,
        dynamic: Bool = false,
        providerMetadata: ProviderMetadata? = nil
    ) {
        self.toolCallId = toolCallId
        self.toolName = toolName
        self.namespace = namespace
        self.input = input
        self.providerExecuted = providerExecuted
        self.dynamic = dynamic
        self.providerMetadata = providerMetadata
    }
}

/// Rich content returned from a client-executed tool.
///
/// Providers that accept structured tool output can retain text and images as
/// content blocks instead of flattening them into a JSON string.
public enum ToolResultContent: Sendable, Hashable {
    case text(String)
    case file(FilePart)
}

/// A tool result, whether received from a provider-executed tool or sent back
/// after a client-executed call.
public struct ToolResult: Sendable, Hashable {
    public var toolCallId: String
    public var toolName: String
    public var result: JSONValue
    /// Structured output when the result contains multimodal content.
    /// `result` remains available for scalar and JSON-valued tool results.
    public var content: [ToolResultContent]?
    public var isError: Bool
    /// Preliminary results replace one another; a final result always follows.
    public var preliminary: Bool
    public var dynamic: Bool
    public var providerMetadata: ProviderMetadata?

    public init(
        toolCallId: String,
        toolName: String,
        result: JSONValue,
        content: [ToolResultContent]? = nil,
        isError: Bool = false,
        preliminary: Bool = false,
        dynamic: Bool = false,
        providerMetadata: ProviderMetadata? = nil
    ) {
        self.toolCallId = toolCallId
        self.toolName = toolName
        self.result = result
        self.content = content
        self.isError = isError
        self.preliminary = preliminary
        self.dynamic = dynamic
        self.providerMetadata = providerMetadata
    }
}

/// A provider event whose wire payload is known but has no provider-neutral
/// event of its own. This is distinct from ``StreamPart/raw(_:)``, which is
/// reserved for event types the mapper does not recognize yet.
public struct ProviderEvent: Sendable, Hashable {
    public var provider: String
    public var type: String
    public var payload: JSONValue

    public init(provider: String, type: String, payload: JSONValue) {
        self.provider = provider
        self.type = type
        self.payload = payload
    }
}

/// A source the model consulted while generating.
public struct Source: Sendable, Hashable {
    public enum Kind: Sendable, Hashable {
        case url(String)
        case document(mediaType: String, filename: String?)
    }

    public var id: String
    public var kind: Kind
    public var title: String?
    public var providerMetadata: ProviderMetadata?

    public init(id: String, kind: Kind, title: String? = nil, providerMetadata: ProviderMetadata? = nil) {
        self.id = id
        self.kind = kind
        self.title = title
        self.providerMetadata = providerMetadata
    }
}

/// A file the model generated.
public struct GeneratedFile: Sendable, Hashable {
    public enum Payload: Sendable, Hashable {
        /// Base64-encoded content, kept as the provider sent it.
        case base64(String)
        case url(String)
    }

    public var mediaType: String
    public var data: Payload
    public var providerMetadata: ProviderMetadata?

    public init(mediaType: String, data: Payload, providerMetadata: ProviderMetadata? = nil) {
        self.mediaType = mediaType
        self.data = data
        self.providerMetadata = providerMetadata
    }
}

/// An error delivered *inside* the stream.
///
/// A stream can carry several of these without ending, so this is a value on
/// the stream rather than a thrown error.
public struct StreamError: Sendable, Hashable, Error {
    public var type: String?
    public var message: String
    public var raw: JSONValue?

    public init(type: String? = nil, message: String, raw: JSONValue? = nil) {
        self.type = type
        self.message = message
        self.raw = raw
    }
}

/// One normalized event in a model response stream.
///
/// This is the whole point of AIKit: every provider's wire format is mapped
/// onto this single enum, so callers are written once instead of once per
/// vendor. The shape follows the AI SDK's `LanguageModelV4StreamPart`, which is
/// the most battle-tested normalization of this problem in the ecosystem.
///
/// Text, reasoning, and tool input each arrive as a `start` → `delta`* → `end`
/// triad keyed by an `id`, so interleaved blocks can be reassembled.
public enum StreamPart: Sendable, Hashable {
    /// Always first. Carries warnings about the request that was sent.
    case streamStart(warnings: [Warning])

    /// Response identity, emitted as soon as the provider reveals it.
    case responseMetadata(ResponseMetadata)

    case textStart(id: String, providerMetadata: ProviderMetadata? = nil)
    case textDelta(id: String, delta: String, providerMetadata: ProviderMetadata? = nil)
    case textEnd(id: String, providerMetadata: ProviderMetadata? = nil)

    case reasoningStart(id: String, providerMetadata: ProviderMetadata? = nil)
    case reasoningDelta(id: String, delta: String, providerMetadata: ProviderMetadata? = nil)
    case reasoningEnd(id: String, providerMetadata: ProviderMetadata? = nil)

    case toolInputStart(
        id: String,
        toolName: String,
        providerExecuted: Bool = false,
        dynamic: Bool = false,
        title: String? = nil,
        providerMetadata: ProviderMetadata? = nil
    )
    /// A fragment of a tool's arguments. Fragments are partial JSON and are
    /// only valid once concatenated across the whole triad.
    case toolInputDelta(id: String, delta: String, providerMetadata: ProviderMetadata? = nil)
    case toolInputEnd(id: String, providerMetadata: ProviderMetadata? = nil)

    case toolCall(ToolCall)
    case toolResult(ToolResult)

    case source(Source)
    case file(GeneratedFile)

    /// Always last on a stream that completed. Carries final usage.
    case finish(usage: Usage, finishReason: FinishReason, providerMetadata: ProviderMetadata? = nil)

    /// A recognized provider lifecycle payload retained byte-for-byte at the
    /// decoded JSON level, including events with no normalized equivalent.
    case providerEvent(ProviderEvent)

    /// An unrecognized provider chunk, passed through untouched. Emitting
    /// `raw` rather than dropping the chunk means a provider can ship a new
    /// block type without this library losing data.
    case raw(JSONValue)

    case error(StreamError)
}
