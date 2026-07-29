import Foundation

/// Maps the Anthropic Messages API streaming protocol onto ``StreamPart``.
///
/// This is a *protocol* implementation, not a provider implementation. Every
/// vendor that speaks the Anthropic Messages wire format reuses it; a provider
/// is then just configuration (base URL, auth, model list) pointing at this
/// mapper. That split is what keeps the untestable surface small — there are
/// far fewer wire formats than there are providers.
///
/// The mapper is a synchronous state machine over already-decoded chunks. It
/// does no I/O, which is what makes it testable against recorded fixtures
/// without any API key.
///
/// - Note: Chunk types with no normalized equivalent yet (server tool use,
///   MCP, code execution results, citations) are surfaced as ``StreamPart/raw(_:)``
///   rather than dropped, so no data is lost while coverage grows.
public struct AnthropicMessagesWire: WireMapper {

    /// What an open content block is accumulating.
    private enum Block: Sendable {
        case text
        case reasoning
        case toolCall(id: String, name: String, input: String)
    }

    /// Open content blocks, keyed by the provider's block index.
    private var blocks: [Int: Block] = [:]
    /// The type string of the block currently open.
    ///
    /// Needed because `signature_delta` is only meaningful on a thinking block
    /// and must be ignored elsewhere.
    private var openBlockType: String?

    /// Usage accumulates across `message_start` and `message_delta`; only the
    /// merged view is correct, so it is held until `message_stop`.
    private var usage: [String: JSONValue] = [:]
    private var finishReason: FinishReason = .other
    private var stopSequence: JSONValue?
    private var stopDetails: JSONValue?
    private var emittedStreamStart = false
    private var emittedFinish = false

    public init() {}

    // MARK: - Entry points

    /// Maps one decoded chunk to zero or more normalized parts.
    ///
    /// A single provider chunk can produce several parts (or none — `ping`
    /// produces nothing), so the return is always a list.
    public mutating func map(chunk: JSONValue) -> [StreamPart] {
        var parts: [StreamPart] = []

        // `stream-start` is contractually first. Emitting it lazily means the
        // caller cannot forget to.
        if !emittedStreamStart {
            emittedStreamStart = true
            parts.append(.streamStart(warnings: []))
        }

        parts.append(contentsOf: mapChunk(chunk))
        return parts
    }

    /// Maps one raw SSE `data:` payload.
    ///
    /// A malformed payload becomes a ``StreamPart/error(_:)`` on the stream
    /// rather than a thrown error, because one bad chunk should not tear down
    /// a response that is otherwise fine.
    public mutating func map(rawJSON: String) -> [StreamPart] {
        // Anthropic never sends this, but the SSE convention is that `[DONE]`
        // terminates a stream. Tolerating it costs nothing and makes this
        // mapper reusable for Anthropic-compatible servers that do send it.
        guard rawJSON.trimmingCharacters(in: .whitespacesAndNewlines) != "[DONE]" else {
            return []
        }

        do {
            return map(chunk: try JSONValue.decode(from: rawJSON))
        } catch {
            return [.error(StreamError(
                type: "parse_error",
                message: "Failed to decode chunk: \(error)",
                raw: .string(rawJSON)
            ))]
        }
    }

    // MARK: - Chunk dispatch

    private mutating func mapChunk(_ chunk: JSONValue) -> [StreamPart] {
        switch chunk["type"]?.stringValue {
        case "ping":
            // Keep-alive only.
            return []

        case "message_start":
            return messageStart(chunk)

        case "content_block_start":
            return contentBlockStart(chunk)

        case "content_block_delta":
            return contentBlockDelta(chunk)

        case "content_block_stop":
            return contentBlockStop(chunk)

        case "message_delta":
            return messageDelta(chunk)

        case "message_stop":
            return messageStop()

        case "error":
            let error = chunk["error"]
            return [.error(StreamError(
                type: error?["type"]?.stringValue,
                message: error?["message"]?.stringValue ?? "Unknown provider error",
                raw: chunk
            ))]

        default:
            return [.raw(chunk)]
        }
    }

    // MARK: - Message lifecycle

    private mutating func messageStart(_ chunk: JSONValue) -> [StreamPart] {
        guard let message = chunk["message"] else { return [.raw(chunk)] }

        if let messageUsage = message["usage"]?.objectValue {
            usage.merge(messageUsage) { _, new in new }
        }

        // A stop reason this early means the turn ended before producing
        // output — a pre-output refusal, most often.
        if let stop = message["stop_reason"]?.stringValue {
            finishReason = Self.mapStopReason(stop)
        }

        return [.responseMetadata(ResponseMetadata(
            id: message["id"]?.stringValue,
            // The serving model, which a server-side fallback can change
            // mid-request. Trust this over the model you asked for.
            modelId: message["model"]?.stringValue
        ))]
    }

    private mutating func messageDelta(_ chunk: JSONValue) -> [StreamPart] {
        if let deltaUsage = chunk["usage"]?.objectValue {
            // Merge rather than replace: `message_delta` carries the output
            // counts, while the input and cache counts arrived at
            // `message_start` and are not repeated.
            usage.merge(deltaUsage) { _, new in new }
        }

        if let delta = chunk["delta"] {
            finishReason = Self.mapStopReason(delta["stop_reason"]?.stringValue)
            stopSequence = delta["stop_sequence"]
            stopDetails = delta["stop_details"]
        }

        return []
    }

    /// Closes the stream if the provider never sent `message_stop`.
    ///
    /// Anthropic emits the terminal `finish` from `message_stop`, so this is
    /// normally a no-op — but a dropped connection would otherwise lose the
    /// final usage entirely.
    public mutating func finish() -> [StreamPart] {
        guard !emittedFinish else { return [] }
        return messageStop()
    }

    private mutating func messageStop() -> [StreamPart] {
        guard !emittedFinish else { return [] }
        emittedFinish = true

        var metadata: [String: JSONValue] = ["usage": .object(usage)]
        if let stopSequence, !stopSequence.isNull {
            metadata["stopSequence"] = stopSequence
        }
        // Only populated on a refusal, and `null` otherwise — see the Messages
        // API docs. Carrying it lets callers read the refusal category.
        if let stopDetails, !stopDetails.isNull {
            metadata["stopDetails"] = stopDetails
        }

        return [.finish(
            usage: AnthropicUsageConverter.convert(.object(usage)),
            finishReason: finishReason,
            providerMetadata: ["anthropic": metadata]
        )]
    }

    // MARK: - Content blocks

    private mutating func contentBlockStart(_ chunk: JSONValue) -> [StreamPart] {
        guard let index = chunk["index"]?.intValue,
              let block = chunk["content_block"] else {
            return [.raw(chunk)]
        }

        let blockType = block["type"]?.stringValue

        // A `fallback` block marks the point where a server-side fallback
        // switched models. There is no normalized primitive for a model hop,
        // and the hop stays visible through `usage.iterations`, so it is
        // passed through raw rather than faked into a content event.
        if blockType == "fallback" {
            return [.raw(chunk)]
        }

        openBlockType = blockType

        switch blockType {
        case "text":
            blocks[index] = .text
            return [.textStart(id: String(index))]

        case "thinking":
            blocks[index] = .reasoning
            return [.reasoningStart(id: String(index))]

        case "redacted_thinking":
            blocks[index] = .reasoning
            // The payload is opaque and must be replayed verbatim on the next
            // turn, so it is preserved rather than discarded.
            var metadata: [String: JSONValue] = [:]
            if let data = block["data"] {
                metadata["redactedData"] = data
            }
            return [.reasoningStart(
                id: String(index),
                providerMetadata: metadata.isEmpty ? nil : ["anthropic": metadata]
            )]

        case "compaction":
            // Server-side compaction arrives as its own block type but is
            // textual. It is normalized as text and tagged, so a caller that
            // cares can tell a summary from the model's own prose.
            blocks[index] = .text
            return [.textStart(
                id: String(index),
                providerMetadata: ["anthropic": ["type": .string("compaction")]]
            )]

        case "tool_use":
            guard let toolId = block["id"]?.stringValue,
                  let toolName = block["name"]?.stringValue else {
                return [.raw(chunk)]
            }

            // Input usually streams in as deltas, but a deferred tool call can
            // carry it inline here. An empty `{}` means "deltas are coming".
            var initialInput = ""
            if let input = block["input"]?.objectValue, !input.isEmpty {
                initialInput = (try? JSONValue.object(input).encodedString()) ?? ""
            }

            blocks[index] = .toolCall(id: toolId, name: toolName, input: initialInput)
            return [.toolInputStart(id: toolId, toolName: toolName)]

        default:
            // server_tool_use, mcp_tool_use, *_tool_result, and anything the
            // provider ships next.
            return [.raw(chunk)]
        }
    }

    private mutating func contentBlockDelta(_ chunk: JSONValue) -> [StreamPart] {
        guard let index = chunk["index"]?.intValue,
              let delta = chunk["delta"] else {
            return [.raw(chunk)]
        }

        switch delta["type"]?.stringValue {
        case "text_delta":
            guard let text = delta["text"]?.stringValue else { return [] }
            return [.textDelta(id: String(index), delta: text)]

        case "thinking_delta":
            guard let thinking = delta["thinking"]?.stringValue else { return [] }
            return [.reasoningDelta(id: String(index), delta: thinking)]

        case "signature_delta":
            // Signatures appear only on thinking blocks. The signature is
            // carried as metadata on an empty delta because it must be echoed
            // back unchanged on the next turn or the API rejects the request.
            guard openBlockType == "thinking",
                  let signature = delta["signature"] else { return [] }
            return [.reasoningDelta(
                id: String(index),
                delta: "",
                providerMetadata: ["anthropic": ["signature": signature]]
            )]

        case "compaction_delta":
            guard let content = delta["content"]?.stringValue else { return [] }
            return [.textDelta(id: String(index), delta: content)]

        case "input_json_delta":
            guard let fragment = delta["partial_json"]?.stringValue,
                  !fragment.isEmpty,
                  case .toolCall(let id, let name, let input)? = blocks[index] else {
                return []
            }
            // Accumulate so `content_block_stop` can emit the assembled call.
            // Fragments are not individually valid JSON.
            blocks[index] = .toolCall(id: id, name: name, input: input + fragment)
            return [.toolInputDelta(id: id, delta: fragment)]

        default:
            // citations_delta and future delta types.
            return [.raw(chunk)]
        }
    }

    private mutating func contentBlockStop(_ chunk: JSONValue) -> [StreamPart] {
        defer { openBlockType = nil }

        guard let index = chunk["index"]?.intValue,
              let block = blocks.removeValue(forKey: index) else {
            return []
        }

        switch block {
        case .text:
            return [.textEnd(id: String(index))]

        case .reasoning:
            return [.reasoningEnd(id: String(index))]

        case .toolCall(let id, let name, let input):
            return [
                .toolInputEnd(id: id),
                .toolCall(ToolCall(
                    toolCallId: id,
                    toolName: name,
                    // A tool with no arguments streams no deltas at all; the
                    // assembled input must still be valid JSON.
                    input: input.isEmpty ? "{}" : input
                )),
            ]
        }
    }

    // MARK: - Stop reasons

    /// Maps Anthropic stop reasons onto the normalized set.
    ///
    /// - SeeAlso: https://docs.anthropic.com/en/api/messages#response-stop-reason
    static func mapStopReason(_ raw: String?) -> FinishReason {
        let unified: FinishReason.Unified
        switch raw {
        case "end_turn", "stop_sequence", "pause_turn":
            unified = .stop
        case "tool_use":
            unified = .toolCalls
        case "max_tokens", "model_context_window_exceeded":
            unified = .length
        case "refusal":
            // A safety classifier declined. The response is a normal HTTP 200
            // with empty or partial content, so callers must branch on this
            // before reading content.
            unified = .contentFilter
        case "compaction":
            unified = .other
        default:
            unified = .other
        }
        return FinishReason(unified: unified, raw: raw)
    }
}
