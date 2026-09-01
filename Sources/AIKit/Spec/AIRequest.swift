import Foundation

/// A provider-neutral model request.
///
/// Unlike ``CallOptions``, which remains the direct ``AIClient`` request shape,
/// this request carries a complete ``ModelDestination``. The facade can
/// therefore switch providers per call without asking callers to rebuild a
/// provider-bound client or specify the model twice.
public struct AIRequest: Sendable {
    public var destination: ModelDestination
    public var prompt: Prompt
    public var maxOutputTokens: Int?
    public var tools: [ToolDefinition]
    public var toolChoice: ToolChoice?
    public var stopSequences: [String]
    public var temperature: Double?
    public var topP: Double?
    public var topK: Int?
    public var responseFormat: JSONValue?
    public var thinking: Thinking?

    /// Provider-specific request fields, namespaced by provider id.
    ///
    /// Only the namespace matching `destination.providerId` is considered by
    /// the selected wire encoder. Options for every other provider are ignored.
    public var providerOptions: ProviderMetadata

    public init(
        destination: ModelDestination,
        prompt: Prompt,
        maxOutputTokens: Int? = nil,
        tools: [ToolDefinition] = [],
        toolChoice: ToolChoice? = nil,
        stopSequences: [String] = [],
        temperature: Double? = nil,
        topP: Double? = nil,
        topK: Int? = nil,
        responseFormat: JSONValue? = nil,
        thinking: Thinking? = nil,
        providerOptions: ProviderMetadata = [:]
    ) {
        self.destination = destination
        self.prompt = prompt
        self.maxOutputTokens = maxOutputTokens
        self.tools = tools
        self.toolChoice = toolChoice
        self.stopSequences = stopSequences
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
        self.responseFormat = responseFormat
        self.thinking = thinking
        self.providerOptions = providerOptions
    }

    /// The existing direct-client request shape used after destination
    /// resolution. Keeping this conversion at the facade boundary makes every
    /// transport continue through ``AIClient`` and the existing wire encoders.
    var callOptions: CallOptions {
        CallOptions(
            model: destination.modelId,
            prompt: prompt,
            maxOutputTokens: maxOutputTokens,
            tools: tools,
            toolChoice: toolChoice,
            stopSequences: stopSequences,
            temperature: temperature,
            topP: topP,
            topK: topK,
            responseFormat: responseFormat,
            thinking: thinking,
            providerOptions: providerOptions
        )
    }
}
