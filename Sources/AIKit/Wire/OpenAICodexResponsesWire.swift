import Foundation

/// ChatGPT Codex's Responses stream dialect.
///
/// Most lifecycle events are byte-for-byte Responses events, so the ordinary
/// mapper remains the normalization engine. Codex's terminal `response.done`
/// alias and provider identity are handled here without rewriting the payload
/// retained for exact history replay.
public struct OpenAICodexResponsesWire: WireMapper {
    private static let terminalAliases: Set<String> = [
        "response.done", "response.completed", "response.incomplete",
    ]
    private static let responseStatuses: Set<String> = [
        "completed", "incomplete", "failed", "cancelled", "queued", "in_progress",
    ]
    private static let transportTerminalEvents: Set<String> = [
        "error", "response.done", "response.completed", "response.incomplete", "response.failed",
    ]

    private var responses = OpenAIResponsesWire()
    private var sawTerminalEvent = false
    private(set) var shouldTerminateTransport = false

    public init() {}

    public mutating func map(chunk: JSONValue) -> [StreamPart] {
        guard let type = chunk["type"]?.stringValue else {
            return relabel(responses.map(chunk: chunk), original: chunk, type: nil)
        }

        if Self.transportTerminalEvents.contains(type) {
            sawTerminalEvent = true
            shouldTerminateTransport = true
        }

        if Self.terminalAliases.contains(type) {
            return relabel(
                responses.mapTerminalAlias(chunk, allowedStatuses: Self.responseStatuses),
                original: chunk,
                type: type
            )
        }
        return relabel(responses.map(chunk: chunk), original: chunk, type: type)
    }

    public mutating func map(rawJSON: String) -> [StreamPart] {
        let trimmed = rawJSON.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed != "[DONE]" else {
            shouldTerminateTransport = true
            return finish()
        }
        do {
            return map(chunk: try JSONValue.decode(from: trimmed))
        } catch {
            return [.error(StreamError(
                type: "parse_error",
                message: "Failed to decode Codex chunk: \(error)",
                raw: .string(rawJSON)
            ))]
        }
    }

    public mutating func finish() -> [StreamPart] {
        var parts = responses.finish()
        guard !sawTerminalEvent, !parts.isEmpty else { return parts }

        let error = StreamPart.error(StreamError(
            type: "incomplete_stream",
            message: "OpenAI Codex Responses stream ended before a terminal response event"
        ))
        if let finishIndex = parts.firstIndex(where: {
            if case .finish = $0 { return true }
            return false
        }) {
            parts.insert(error, at: finishIndex)
        } else {
            parts.append(error)
        }
        return parts
    }

    private func relabel(
        _ parts: [StreamPart],
        original: JSONValue,
        type: String?
    ) -> [StreamPart] {
        parts.map { part in
            switch part {
            case .providerEvent:
                return .providerEvent(ProviderEvent(
                    provider: "openai-codex",
                    type: type ?? "event",
                    payload: original
                ))
            case .error(var error):
                if error.type == "malformed_event" {
                    error.message = error.message.replacingOccurrences(
                        of: "OpenAI Responses",
                        with: "OpenAI Codex Responses"
                    )
                }
                // Errors remain attributable to the bytes the Codex backend
                // actually sent, never an internal shared-mapper projection.
                if error.raw != nil { error.raw = original }
                return .error(error)
            default:
                return part
            }
        }
    }
}
