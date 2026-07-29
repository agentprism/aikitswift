import Foundation

/// Everything needed to make one model call.
///
/// Provider-specific knobs live in ``providerOptions`` rather than growing a
/// field per vendor. Settings a provider cannot honour are dropped and reported
/// as a ``Warning`` on ``StreamPart/streamStart(warnings:)`` — the call still
/// runs, but the caller is told what was ignored instead of silently getting
/// different behaviour.
public struct CallOptions: Sendable {
    /// The model id, in the provider's own naming.
    public var model: String
    public var prompt: Prompt

    /// Hard cap on output tokens.
    ///
    /// On models where thinking is enabled this budget covers reasoning *and*
    /// the visible answer, so a value tuned for a non-thinking model can
    /// truncate the response mid-sentence.
    public var maxOutputTokens: Int?

    public var tools: [ToolDefinition]
    public var toolChoice: ToolChoice?
    public var stopSequences: [String]

    /// Sampling temperature.
    ///
    /// Newer Anthropic models reject this outright; the Anthropic encoder drops
    /// it and emits a warning rather than letting the request fail.
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?

    /// A JSON Schema the response must conform to, where supported.
    public var responseFormat: JSONValue?

    /// Escape hatch for anything vendor-specific, namespaced by provider id.
    ///
    /// Merged into the request body by that provider's encoder and ignored by
    /// every other, so one `CallOptions` can carry settings for several
    /// providers at once.
    public var providerOptions: ProviderMetadata

    public init(
        model: String,
        prompt: Prompt,
        maxOutputTokens: Int? = nil,
        tools: [ToolDefinition] = [],
        toolChoice: ToolChoice? = nil,
        stopSequences: [String] = [],
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        responseFormat: JSONValue? = nil,
        providerOptions: ProviderMetadata = [:]
    ) {
        self.model = model
        self.prompt = prompt
        self.maxOutputTokens = maxOutputTokens
        self.tools = tools
        self.toolChoice = toolChoice
        self.stopSequences = stopSequences
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.responseFormat = responseFormat
        self.providerOptions = providerOptions
    }

    /// Options this provider should merge into the request body.
    public func options(for providerId: String) -> [String: JSONValue] {
        providerOptions[providerId] ?? [:]
    }
}

/// The outcome of encoding ``CallOptions`` into a provider request body.
///
/// Warnings ride along with the body so that a dropped setting surfaces to the
/// caller on the stream instead of disappearing.
public struct EncodedRequest: Sendable {
    public var body: JSONValue
    public var warnings: [Warning]

    public init(body: JSONValue, warnings: [Warning] = []) {
        self.body = body
        self.warnings = warnings
    }
}
