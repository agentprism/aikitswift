import Foundation

/// Maps the OpenAI Chat Completions streaming protocol onto ``StreamPart``.
///
/// The highest-leverage protocol in the library: of the fifty providers in the
/// catalog, thirty-eight speak this one. Getting it right covers three quarters
/// of the ecosystem, including DeepSeek, Moonshot, Zhipu, MiniMax, SiliconFlow,
/// Groq, Together, Fireworks, OpenRouter and every "OpenAI-compatible" endpoint.
///
/// Vendors extend it inconsistently — reasoning in particular has at least
/// three spellings in the wild — so the mapper accepts all of them rather than
/// forcing each vendor to become its own protocol.
public struct OpenAICompletionsWire: WireMapper {

    /// A tool call being assembled across chunks.
    ///
    /// Keyed by the provider's `tool_calls[].index`, because after the first
    /// chunk the id and name are omitted and only the index identifies which
    /// call an argument fragment belongs to.
    private struct PendingToolCall {
        var id: String
        var name: String
        var arguments: String
    }

    private var pendingToolCalls: [Int: PendingToolCall] = [:]
    /// Preserves call order; dictionary iteration order is not stable.
    private var toolCallOrder: [Int] = []

    private var textOpen: Set<String> = []
    private var reasoningOpen: Set<String> = []

    private var finishReason: FinishReason = .other
    private var usage: JSONValue?
    private var emittedStreamStart = false
    private var responseId: String?
    private var responseTimestamp: Date?
    private var responseModelId: String?
    private var finished = false

    public init() {}

    // MARK: - Entry points

    public mutating func map(chunk: JSONValue) -> [StreamPart] {
        var parts: [StreamPart] = []

        if !emittedStreamStart {
            emittedStreamStart = true
            parts.append(.streamStart(warnings: []))
        }

        parts.append(contentsOf: mapChunk(chunk))
        return parts
    }

    /// Maps one raw SSE `data:` payload.
    ///
    /// `[DONE]` terminates an OpenAI stream and is where the terminal `finish`
    /// is emitted, since the protocol has no dedicated end event.
    public mutating func map(rawJSON: String) -> [StreamPart] {
        let trimmed = rawJSON.trimmingCharacters(in: .whitespacesAndNewlines)

        if trimmed == "[DONE]" {
            return finish()
        }

        do {
            return map(chunk: try JSONValue.decode(from: trimmed))
        } catch {
            return [.error(StreamError(
                type: "parse_error",
                message: "Failed to decode chunk: \(error)",
                raw: .string(rawJSON)
            ))]
        }
    }

    /// Closes the stream.
    ///
    /// Call this when the connection ends without a `[DONE]` sentinel — several
    /// compatible servers simply stop sending. Without it the terminal `finish`
    /// (and the usage it carries) would be lost.
    public mutating func finish() -> [StreamPart] {
        guard !finished else { return [] }
        finished = true

        var parts: [StreamPart] = []

        if !emittedStreamStart {
            emittedStreamStart = true
            parts.insert(.streamStart(warnings: []), at: 0)
        }

        // Close anything the provider left open. A stream cut short mid-block
        // would otherwise produce an unbalanced triad.
        parts.append(contentsOf: closeOpenBlocks())
        parts.append(contentsOf: flushPendingToolCalls())

        parts.append(.finish(
            usage: usage.map(Self.convertUsage) ?? .empty,
            finishReason: finishReason,
            providerMetadata: usage.map { ["openai": ["usage": $0]] }
        ))

        return parts
    }

    // MARK: - Chunk handling

    private mutating func mapChunk(_ chunk: JSONValue) -> [StreamPart] {
        var parts: [StreamPart] = []

        // Usage arrives on its own trailing chunk when the caller opted in via
        // `stream_options.include_usage`, and is absent otherwise.
        if let chunkUsage = chunk["usage"], !chunkUsage.isNull {
            usage = chunkUsage
        }

        if let metadata = captureResponseMetadata(from: chunk) {
            parts.append(.responseMetadata(metadata))
        }

        guard let choices = chunk["choices"]?.arrayValue, !choices.isEmpty else {
            // A chunk with no choices is usually the usage-only trailer or a
            // content-filter preamble. It carried no content, so nothing more
            // to emit — but it was not ignored, since usage was captured above.
            return parts
        }

        for choice in choices {
            parts.append(contentsOf: mapChoice(choice))
        }

        return parts
    }

    /// Providers do not all reveal response fields in the same chunk. Retain
    /// each field the first time it appears and emit an update when later
    /// chunks add information, especially the actual serving model.
    private mutating func captureResponseMetadata(from chunk: JSONValue) -> ResponseMetadata? {
        var changed = false

        if responseId == nil, let id = chunk["id"]?.stringValue, !id.isEmpty {
            responseId = id
            changed = true
        }
        if responseTimestamp == nil, let created = chunk["created"]?.doubleValue {
            responseTimestamp = Date(timeIntervalSince1970: created)
            changed = true
        }
        if responseModelId == nil, let model = chunk["model"]?.stringValue, !model.isEmpty {
            responseModelId = model
            changed = true
        }

        guard changed else { return nil }
        return ResponseMetadata(
            id: responseId,
            timestamp: responseTimestamp,
            modelId: responseModelId
        )
    }

    private mutating func mapChoice(_ choice: JSONValue) -> [StreamPart] {
        var parts: [StreamPart] = []
        let index = choice["index"]?.intValue ?? 0
        let id = String(index)

        if let delta = choice["delta"] {
            parts.append(contentsOf: mapReasoning(delta, id: id))
            parts.append(contentsOf: mapContent(delta, id: id))
            parts.append(contentsOf: mapToolCalls(delta))
        }

        if let reason = choice["finish_reason"]?.stringValue {
            finishReason = Self.mapFinishReason(reason)

            // `finish_reason` marks the end of generation for this choice, so
            // any open block closes here. Tool calls are flushed now because a
            // provider that never sends `[DONE]` would otherwise drop them.
            parts.append(contentsOf: closeOpenBlocks())
            parts.append(contentsOf: flushPendingToolCalls())
        }

        return parts
    }

    /// Handles the several spellings of streamed reasoning.
    ///
    /// DeepSeek uses `reasoning_content`; others use `reasoning`, sometimes as
    /// a bare string and sometimes as an object with a `text` field. All three
    /// are the same concept, so all three are accepted.
    private mutating func mapReasoning(_ delta: JSONValue, id: String) -> [StreamPart] {
        let text: String?
        if let value = delta["reasoning_content"]?.stringValue {
            text = value
        } else if let value = delta["reasoning"]?.stringValue {
            text = value
        } else if let value = delta["reasoning"]?["text"]?.stringValue {
            text = value
        } else {
            text = nil
        }

        guard let text, !text.isEmpty else { return [] }

        var parts: [StreamPart] = []
        if !reasoningOpen.contains(id) {
            reasoningOpen.insert(id)
            parts.append(.reasoningStart(id: id))
        }
        parts.append(.reasoningDelta(id: id, delta: text))
        return parts
    }

    private mutating func mapContent(_ delta: JSONValue, id: String) -> [StreamPart] {
        guard let content = delta["content"]?.stringValue, !content.isEmpty else { return [] }

        var parts: [StreamPart] = []

        // Reasoning always precedes visible content, so an open reasoning block
        // closes as soon as the answer starts.
        if reasoningOpen.contains(id) {
            reasoningOpen.remove(id)
            parts.append(.reasoningEnd(id: id))
        }

        if !textOpen.contains(id) {
            textOpen.insert(id)
            parts.append(.textStart(id: id))
        }
        parts.append(.textDelta(id: id, delta: content))
        return parts
    }

    private mutating func mapToolCalls(_ delta: JSONValue) -> [StreamPart] {
        guard let calls = delta["tool_calls"]?.arrayValue else { return [] }

        var parts: [StreamPart] = []

        for call in calls {
            let index = call["index"]?.intValue ?? 0
            let function = call["function"]

            if pendingToolCalls[index] == nil {
                // First chunk for this slot: it carries the id and name, which
                // later fragments omit.
                let callId = call["id"]?.stringValue ?? "call_\(index)"
                let name = function?["name"]?.stringValue ?? ""

                pendingToolCalls[index] = PendingToolCall(id: callId, name: name, arguments: "")
                toolCallOrder.append(index)
                parts.append(.toolInputStart(id: callId, toolName: name))
            } else if let name = function?["name"]?.stringValue, !name.isEmpty,
                      pendingToolCalls[index]?.name.isEmpty == true {
                // Some servers split the name across chunks rather than
                // sending it whole on the first one.
                pendingToolCalls[index]?.name = name
            }

            if let fragment = function?["arguments"]?.stringValue, !fragment.isEmpty,
               let pending = pendingToolCalls[index] {
                pendingToolCalls[index]?.arguments = pending.arguments + fragment
                parts.append(.toolInputDelta(id: pending.id, delta: fragment))
            }
        }

        return parts
    }

    // MARK: - Closing

    private mutating func closeOpenBlocks() -> [StreamPart] {
        var parts: [StreamPart] = []

        for id in reasoningOpen.sorted() {
            parts.append(.reasoningEnd(id: id))
        }
        reasoningOpen.removeAll()

        for id in textOpen.sorted() {
            parts.append(.textEnd(id: id))
        }
        textOpen.removeAll()

        return parts
    }

    private mutating func flushPendingToolCalls() -> [StreamPart] {
        guard !pendingToolCalls.isEmpty else { return [] }

        var parts: [StreamPart] = []

        for index in toolCallOrder {
            guard let pending = pendingToolCalls[index] else { continue }
            parts.append(.toolInputEnd(id: pending.id))
            parts.append(.toolCall(ToolCall(
                toolCallId: pending.id,
                toolName: pending.name,
                // A no-argument tool streams nothing; the assembled input still
                // has to be valid JSON.
                input: pending.arguments.isEmpty ? "{}" : pending.arguments
            )))
        }

        pendingToolCalls.removeAll()
        toolCallOrder.removeAll()

        return parts
    }

    // MARK: - Mapping tables

    static func mapFinishReason(_ raw: String?) -> FinishReason {
        let unified: FinishReason.Unified
        switch raw {
        case "stop": unified = .stop
        case "length": unified = .length
        case "tool_calls", "function_call": unified = .toolCalls
        case "content_filter": unified = .contentFilter
        default: unified = .other
        }
        return FinishReason(unified: unified, raw: raw)
    }

    /// Converts an OpenAI `usage` object into normalized ``Usage``.
    ///
    /// - Important: `prompt_tokens` here **includes** cached tokens, the
    ///   opposite of Anthropic's convention where `input_tokens` excludes them.
    ///   Applying one provider's arithmetic to the other double-counts the
    ///   cache, which is exactly the kind of error that never raises and only
    ///   shows up on a bill.
    static func convertUsage(_ usage: JSONValue) -> Usage {
        let promptTokens = usage["prompt_tokens"]?.intValue
        let completionTokens = usage["completion_tokens"]?.intValue
        let cachedTokens = usage["prompt_tokens_details"]?["cached_tokens"]?.intValue ?? 0
        let reasoningTokens = usage["completion_tokens_details"]?["reasoning_tokens"]?.intValue

        return Usage(
            inputTokens: .init(
                total: promptTokens,
                noCache: promptTokens.map { $0 - cachedTokens },
                cacheRead: cachedTokens,
                // Chat Completions has no notion of a cache write: caching is
                // automatic and unpriced, so there is nothing to report.
                cacheWrite: 0
            ),
            outputTokens: .init(
                total: completionTokens,
                text: {
                    guard let completionTokens, let reasoningTokens else { return nil }
                    return completionTokens - reasoningTokens
                }(),
                reasoning: reasoningTokens
            ),
            raw: usage
        )
    }
}
