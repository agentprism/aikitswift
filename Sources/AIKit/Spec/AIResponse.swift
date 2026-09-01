import Foundation

/// A complete model response, assembled from normalized stream parts.
///
/// This is the aggregate view for callers who want the outcome rather than the
/// events: the full text, the reasoning, the tool calls, the usage. It stores
/// only ``parts`` — every derived property reads from them, so nothing the
/// provider sent is discarded and the derived values cannot drift from the
/// source.
///
/// Both fetch paths produce one: ``AIClient/generate(_:)`` returns it directly,
/// and a stream becomes one via `collect()`:
///
/// ```swift
/// let response = try await client.generate(options)          // complete response
/// let response = try await client.stream(options).collect()  // streaming
/// response.text
/// ```
public struct AIResponse: Sendable, Hashable {
    /// Every part, in arrival order. The complete record; all other properties
    /// are views over this.
    public var parts: [StreamPart]

    public init(parts: [StreamPart]) {
        self.parts = parts
    }

    // MARK: - Aggregated content

    /// All visible text, concatenated across text blocks.
    public var text: String {
        parts.compactMap { if case .textDelta(_, let delta, _) = $0 { delta } else { nil } }
            .joined()
    }

    /// All reasoning, concatenated across thinking blocks.
    public var reasoning: String {
        parts.compactMap { if case .reasoningDelta(_, let delta, _) = $0 { delta } else { nil } }
            .joined()
    }

    /// Every complete tool call, in the order the model made them — including
    /// provider-executed ones, which carry results in ``toolResults`` rather
    /// than waiting on the caller.
    public var toolCalls: [ToolCall] {
        parts.compactMap { if case .toolCall(let call) = $0 { call } else { nil } }
    }

    /// Tool calls the *caller* must execute and answer with a
    /// ``Message/toolResult(toolCallId:toolName:result:isError:)`` message.
    public var pendingToolCalls: [ToolCall] {
        toolCalls.filter { !$0.providerExecuted }
    }

    /// Results of tools the provider executed server-side.
    public var toolResults: [ToolResult] {
        parts.compactMap { if case .toolResult(let result) = $0 { result } else { nil } }
    }

    public var sources: [Source] {
        parts.compactMap { if case .source(let source) = $0 { source } else { nil } }
    }

    public var files: [GeneratedFile] {
        parts.compactMap { if case .file(let file) = $0 { file } else { nil } }
    }

    /// Recognized provider lifecycle payloads in arrival order.
    public var providerEvents: [ProviderEvent] {
        parts.compactMap { if case .providerEvent(let event) = $0 { event } else { nil } }
    }

    // MARK: - Outcome

    /// Warnings about the request — settings that were dropped or adjusted
    /// before sending.
    public var warnings: [Warning] {
        parts.compactMap { if case .streamStart(let warnings) = $0 { warnings } else { nil } }
            .flatMap { $0 }
    }

    /// Errors the provider delivered inside the stream without ending it.
    public var errors: [StreamError] {
        parts.compactMap { if case .error(let error) = $0 { error } else { nil } }
    }

    public var usage: Usage {
        for part in parts.reversed() {
            if case .finish(let usage, _, _) = part { return usage }
        }
        return .empty
    }

    /// Why generation stopped. `nil` when the stream never finished — the
    /// response is incomplete and should be treated as such.
    public var finishReason: FinishReason? {
        for part in parts.reversed() {
            if case .finish(_, let reason, _) = part { return reason }
        }
        return nil
    }

    public var metadata: ResponseMetadata? {
        parts.compactMap { if case .responseMetadata(let metadata) = $0 { metadata } else { nil } }
            .first
    }

    // MARK: - Multi-turn replay

    /// This response as the assistant message for the next turn.
    ///
    /// This is the supported way to continue a conversation. Append it, then
    /// answer any ``pendingToolCalls`` with tool-result messages:
    ///
    /// ```swift
    /// prompt.append(response.assistantMessage)
    /// for call in response.pendingToolCalls {
    ///     prompt.append(.toolResult(
    ///         toolCallId: call.toolCallId,
    ///         toolName: call.toolName,
    ///         result: try await run(call)
    ///     ))
    /// }
    /// ```
    ///
    /// Blocks keep their stream order, and reasoning blocks keep their
    /// `providerMetadata` — some providers sign thinking blocks and reject the
    /// next request unless the signature comes back byte-for-byte, which is why
    /// assembling this message by hand is the classic way multi-turn reasoning
    /// breaks. Provider-executed tool calls remain absent from the normalized
    /// content. OpenAI Responses history is retained separately as its exact
    /// output-item list because those provider-owned items are valid next-turn
    /// input and carry fields the normalized content intentionally does not.
    public var assistantMessage: Message {
        // Blocks are keyed by the triad id so interleaved deltas land in the
        // block that started them, and ordered by when each block started.
        enum Block {
            case text(String)
            case reasoning(String, metadata: ProviderMetadata?)
            case toolCall(ToolCall?)
        }
        var order: [String] = []
        var blocks: [String: Block] = [:]

        func upsert(_ id: String, _ update: (inout Block) -> Void, makeNew: () -> Block) {
            if blocks[id] == nil {
                order.append(id)
                blocks[id] = makeNew()
            }
            update(&blocks[id]!)
        }

        for part in parts {
            switch part {
            case .textStart(let id, _):
                upsert("text:\(id)", { _ in }, makeNew: { .text("") })
            case .textDelta(let id, let delta, _):
                upsert("text:\(id)", {
                    if case .text(let text) = $0 { $0 = .text(text + delta) }
                }, makeNew: { .text("") })

            case .reasoningStart(let id, let metadata):
                upsert("reasoning:\(id)", {
                    if case .reasoning(let text, let existing) = $0 {
                        $0 = .reasoning(text, metadata: merge(existing, metadata))
                    }
                }, makeNew: { .reasoning("", metadata: nil) })
            case .reasoningDelta(let id, let delta, let metadata):
                upsert("reasoning:\(id)", {
                    if case .reasoning(let text, let existing) = $0 {
                        $0 = .reasoning(text + delta, metadata: merge(existing, metadata))
                    }
                }, makeNew: { .reasoning("", metadata: nil) })
            case .reasoningEnd(let id, let metadata):
                upsert("reasoning:\(id)", {
                    if case .reasoning(let text, let existing) = $0 {
                        $0 = .reasoning(text, metadata: merge(existing, metadata))
                    }
                }, makeNew: { .reasoning("", metadata: nil) })

            case .toolInputStart(let id, _, let providerExecuted, _, _, _) where !providerExecuted:
                // Reserve the position; the complete call fills it below.
                upsert("tool:\(id)", { _ in }, makeNew: { .toolCall(nil) })
            case .toolCall(let call) where !call.providerExecuted:
                upsert("tool:\(call.toolCallId)", { $0 = .toolCall(call) }, makeNew: { .toolCall(call) })

            default:
                break
            }
        }

        let content: [ContentPart] = order.compactMap { id in
            switch blocks[id]! {
            case .text(let text):
                return text.isEmpty ? nil : .text(text)
            case .reasoning(let text, let metadata):
                return text.isEmpty && metadata == nil
                    ? nil
                    : .reasoning(text, providerMetadata: metadata)
            case .toolCall(let call):
                return call.map { .toolCall($0) }
            }
        }

        var providerOptions: ProviderMetadata?
        if let outputItems = openAIOutputItems, !outputItems.isEmpty {
            providerOptions = ["openai": ["outputItems": .array(outputItems)]]
        }

        return Message(role: .assistant, content: content, providerOptions: providerOptions)
    }

    /// The terminal Responses payload is authoritative because it contains the
    /// final form of every output item. A partial stream falls back to the last
    /// `output_item.done` payloads it did receive.
    private var openAIOutputItems: [JSONValue]? {
        for event in providerEvents.reversed()
        where ["openai", "openai-codex"].contains(event.provider)
            && ["response.completed", "response.incomplete", "response.failed", "response.done"]
                .contains(event.type) {
            if let output = event.payload["response"]?["output"]?.arrayValue {
                return output
            }
        }

        let completedItems = providerEvents.compactMap { event -> JSONValue? in
            guard ["openai", "openai-codex"].contains(event.provider),
                  event.type == "response.output_item.done"
            else { return nil }
            return event.payload["item"]
        }
        return completedItems.isEmpty ? nil : completedItems
    }
}

/// Namespace-wise merge; later values win within a namespace.
private func merge(_ base: ProviderMetadata?, _ update: ProviderMetadata?) -> ProviderMetadata? {
    guard var merged = base else { return update }
    for (namespace, values) in update ?? [:] {
        merged[namespace, default: [:]].merge(values) { _, new in new }
    }
    return merged
}

// MARK: - Collecting a stream

extension AsyncSequence where Element == StreamPart {
    /// Drains the stream and returns the assembled response.
    ///
    /// For callers who want streaming transport — first token latency, a
    /// provider that only streams — without stream handling:
    ///
    /// ```swift
    /// let response = try await client.stream(options).collect()
    /// ```
    public func collect() async throws -> AIResponse {
        var parts: [StreamPart] = []
        for try await part in self { parts.append(part) }
        return AIResponse(parts: parts)
    }
}
