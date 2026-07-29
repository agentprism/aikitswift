import Foundation

/// Maps the OpenAI Responses API streaming protocol onto ``StreamPart``.
///
/// Structurally the opposite of Chat Completions. Where Completions sends one
/// chunk shape and leaves the caller to infer structure from deltas, Responses
/// sends explicit lifecycle events (`output_item.added`, `content_part.added`,
/// `…done`) that already describe the block structure. Less inference, many
/// more event types.
///
/// Server-side tool activity — code interpreter, shell, web search, MCP, apply
/// patch — is surfaced as ``StreamPart/raw(_:)``: those have no normalized
/// equivalent yet, and passing them through keeps the data available.
public struct OpenAIResponsesWire: WireMapper {

    private struct PendingToolCall {
        var callId: String
        var name: String
        var arguments: String
    }

    /// Function calls keyed by the item id their argument deltas reference.
    private var pendingToolCalls: [String: PendingToolCall] = [:]
    private var toolCallOrder: [String] = []

    private var textOpen: Set<String> = []
    private var reasoningOpen: Set<String> = []

    private var finishReason: FinishReason = .other
    private var usage: JSONValue?
    private var emittedStreamStart = false
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

    public mutating func map(rawJSON: String) -> [StreamPart] {
        let trimmed = rawJSON.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed != "[DONE]" else { return finish() }

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

    public mutating func finish() -> [StreamPart] {
        guard !finished else { return [] }
        finished = true

        var parts: [StreamPart] = []

        if !emittedStreamStart {
            emittedStreamStart = true
            parts.insert(.streamStart(warnings: []), at: 0)
        }

        parts.append(contentsOf: closeOpenBlocks())
        parts.append(contentsOf: flushPendingToolCalls())
        parts.append(.finish(
            usage: usage.map(Self.convertUsage) ?? .empty,
            finishReason: finishReason,
            providerMetadata: usage.map { ["openai": ["usage": $0]] }
        ))

        return parts
    }

    // MARK: - Event dispatch

    private mutating func mapChunk(_ chunk: JSONValue) -> [StreamPart] {
        switch chunk["type"]?.stringValue {
        case "response.created":
            guard let response = chunk["response"] else { return [.raw(chunk)] }
            return [.responseMetadata(ResponseMetadata(
                id: response["id"]?.stringValue,
                timestamp: response["created_at"]?.doubleValue.map { Date(timeIntervalSince1970: $0) },
                modelId: response["model"]?.stringValue
            ))]

        case "response.in_progress":
            // Progress heartbeat; carries nothing new.
            return []

        case "response.output_item.added":
            return outputItemAdded(chunk)

        case "response.output_item.done":
            return outputItemDone(chunk)

        case "response.content_part.added":
            return contentPartAdded(chunk)

        case "response.output_text.delta":
            guard let id = textId(chunk), let delta = chunk["delta"]?.stringValue else { return [] }
            return [.textDelta(id: id, delta: delta)]

        case "response.content_part.done":
            guard let id = textId(chunk), textOpen.remove(id) != nil else { return [] }
            return [.textEnd(id: id)]

        case "response.output_text.done":
            // Redundant with `content_part.done`, which is what closes the
            // block. Emitting on both would end the triad twice.
            return []

        case "response.reasoning_summary_text.delta":
            return reasoningDelta(chunk)

        case "response.function_call_arguments.delta":
            return functionArgumentsDelta(chunk)

        case "response.function_call_arguments.done":
            // The complete arguments are echoed here, but they were already
            // accumulated from the deltas. The call is emitted when its item
            // completes, so ordering stays consistent with other protocols.
            return []

        case "response.completed", "response.incomplete", "response.failed":
            return responseTerminal(chunk)

        case "error":
            return [.error(StreamError(
                type: chunk["code"]?.stringValue,
                message: chunk["message"]?.stringValue ?? "Unknown provider error",
                raw: chunk
            ))]

        default:
            // Server-side tool activity and anything OpenAI adds next.
            return [.raw(chunk)]
        }
    }

    // MARK: - Items

    private mutating func outputItemAdded(_ chunk: JSONValue) -> [StreamPart] {
        guard let item = chunk["item"], let itemId = item["id"]?.stringValue else {
            return [.raw(chunk)]
        }

        switch item["type"]?.stringValue {
        case "function_call":
            // `call_id` is the identifier the tool result must reference; `id`
            // only addresses the streaming item. Confusing the two produces a
            // tool result the model cannot match to its call.
            let callId = item["call_id"]?.stringValue ?? itemId
            let name = item["name"]?.stringValue ?? ""

            pendingToolCalls[itemId] = PendingToolCall(callId: callId, name: name, arguments: "")
            toolCallOrder.append(itemId)
            return [.toolInputStart(id: callId, toolName: name)]

        case "reasoning":
            // Deltas may or may not follow, so the block opens lazily on the
            // first one rather than here.
            return []

        case "message":
            return []

        default:
            return [.raw(chunk)]
        }
    }

    private mutating func outputItemDone(_ chunk: JSONValue) -> [StreamPart] {
        guard let itemId = chunk["item"]?["id"]?.stringValue else { return [] }

        var parts: [StreamPart] = []

        if reasoningOpen.remove(itemId) != nil {
            parts.append(.reasoningEnd(id: itemId))
        }

        if let pending = pendingToolCalls.removeValue(forKey: itemId) {
            toolCallOrder.removeAll { $0 == itemId }
            parts.append(.toolInputEnd(id: pending.callId))
            parts.append(.toolCall(ToolCall(
                toolCallId: pending.callId,
                toolName: pending.name,
                input: pending.arguments.isEmpty ? "{}" : pending.arguments
            )))
        }

        return parts
    }

    private mutating func contentPartAdded(_ chunk: JSONValue) -> [StreamPart] {
        guard chunk["part"]?["type"]?.stringValue == "output_text",
              let id = textId(chunk) else { return [] }

        guard !textOpen.contains(id) else { return [] }
        textOpen.insert(id)
        return [.textStart(id: id)]
    }

    private mutating func reasoningDelta(_ chunk: JSONValue) -> [StreamPart] {
        guard let itemId = chunk["item_id"]?.stringValue,
              let delta = chunk["delta"]?.stringValue, !delta.isEmpty else { return [] }

        var parts: [StreamPart] = []
        if !reasoningOpen.contains(itemId) {
            reasoningOpen.insert(itemId)
            parts.append(.reasoningStart(id: itemId))
        }
        parts.append(.reasoningDelta(id: itemId, delta: delta))
        return parts
    }

    private mutating func functionArgumentsDelta(_ chunk: JSONValue) -> [StreamPart] {
        guard let itemId = chunk["item_id"]?.stringValue,
              let delta = chunk["delta"]?.stringValue, !delta.isEmpty,
              let pending = pendingToolCalls[itemId] else { return [] }

        pendingToolCalls[itemId]?.arguments = pending.arguments + delta
        return [.toolInputDelta(id: pending.callId, delta: delta)]
    }

    private mutating func responseTerminal(_ chunk: JSONValue) -> [StreamPart] {
        guard let response = chunk["response"] else { return [.raw(chunk)] }

        if let responseUsage = response["usage"], !responseUsage.isNull {
            usage = responseUsage
        }

        finishReason = Self.mapStatus(
            response["status"]?.stringValue,
            incompleteReason: response["incomplete_details"]?["reason"]?.stringValue,
            hasToolCalls: !toolCallOrder.isEmpty
        )

        return []
    }

    // MARK: - Helpers

    /// Text blocks are addressed by item *and* content index, since one item
    /// can carry several content parts.
    private func textId(_ chunk: JSONValue) -> String? {
        guard let itemId = chunk["item_id"]?.stringValue else { return nil }
        let contentIndex = chunk["content_index"]?.intValue ?? 0
        return "\(itemId)#\(contentIndex)"
    }

    private mutating func closeOpenBlocks() -> [StreamPart] {
        var parts: [StreamPart] = []

        for id in reasoningOpen.sorted() { parts.append(.reasoningEnd(id: id)) }
        reasoningOpen.removeAll()

        for id in textOpen.sorted() { parts.append(.textEnd(id: id)) }
        textOpen.removeAll()

        return parts
    }

    private mutating func flushPendingToolCalls() -> [StreamPart] {
        guard !pendingToolCalls.isEmpty else { return [] }

        var parts: [StreamPart] = []
        for itemId in toolCallOrder {
            guard let pending = pendingToolCalls[itemId] else { continue }
            parts.append(.toolInputEnd(id: pending.callId))
            parts.append(.toolCall(ToolCall(
                toolCallId: pending.callId,
                toolName: pending.name,
                input: pending.arguments.isEmpty ? "{}" : pending.arguments
            )))
        }

        pendingToolCalls.removeAll()
        toolCallOrder.removeAll()
        return parts
    }

    // MARK: - Mapping tables

    static func mapStatus(
        _ status: String?,
        incompleteReason: String? = nil,
        hasToolCalls: Bool = false
    ) -> FinishReason {
        let unified: FinishReason.Unified
        switch status {
        case "completed":
            // The Responses API reports success without saying whether tools
            // are pending, so the caller's only signal is the call itself.
            unified = hasToolCalls ? .toolCalls : .stop
        case "incomplete":
            unified = incompleteReason == "max_output_tokens" ? .length : .contentFilter
        case "failed":
            unified = .error
        default:
            unified = .other
        }
        return FinishReason(unified: unified, raw: status)
    }

    /// - Important: like Chat Completions and unlike Anthropic, `input_tokens`
    ///   here **includes** cached tokens, and `output_tokens` **includes**
    ///   reasoning tokens.
    static func convertUsage(_ usage: JSONValue) -> Usage {
        let inputTokens = usage["input_tokens"]?.intValue
        let outputTokens = usage["output_tokens"]?.intValue
        let cachedTokens = usage["input_tokens_details"]?["cached_tokens"]?.intValue ?? 0
        let reasoningTokens = usage["output_tokens_details"]?["reasoning_tokens"]?.intValue

        return Usage(
            inputTokens: .init(
                total: inputTokens,
                noCache: inputTokens.map { $0 - cachedTokens },
                cacheRead: cachedTokens,
                cacheWrite: 0
            ),
            outputTokens: .init(
                total: outputTokens,
                text: {
                    guard let outputTokens, let reasoningTokens else { return nil }
                    return outputTokens - reasoningTokens
                }(),
                reasoning: reasoningTokens
            ),
            raw: usage
        )
    }
}
