import Foundation
import Testing

@testable import AIKit

/// Invariants every wire protocol must satisfy, regardless of vendor.
///
/// Sharing one checker across protocols is deliberate. These are properties of
/// the *normalized* stream, so a mapper that violates one is broken no matter
/// which API it speaks — and a new protocol inherits the whole suite for free.
enum WireConformance {

    /// Runs every invariant against one recording's replayed streams.
    static func check(_ streams: [[StreamPart]], label: String) {
        for (index, parts) in streams.enumerated() {
            let name = streams.count == 1 ? label : "\(label)[\(index)]"

            checkStreamStart(parts, name: name)
            checkFinish(parts, name: name)
            checkBalancedTriads(parts, name: name)
            checkToolCalls(parts, name: name)
            checkUsage(parts, name: name)
            checkNoParseErrors(parts, name: name)
        }
    }

    /// `stream-start` is contractually first, so callers can rely on it to
    /// surface request warnings before any content arrives.
    private static func checkStreamStart(_ parts: [StreamPart], name: String) {
        guard case .streamStart = parts.first else {
            Issue.record("\(name): expected stream-start first, got \(String(describing: parts.first))")
            return
        }
    }

    /// Exactly one terminal `finish`, and nothing after it.
    ///
    /// This is what makes final usage reliably reachable: a caller that reads
    /// usage off `finish` must never have to wonder whether another is coming.
    private static func checkFinish(_ parts: [StreamPart], name: String) {
        let indices = parts.indices.filter {
            if case .finish = parts[$0] { return true }
            return false
        }

        #expect(indices.count == 1, "\(name): expected one finish, found \(indices.count)")
        #expect(indices.last == parts.count - 1, "\(name): finish is not the last part")
    }

    /// Text, reasoning and tool-input all arrive as `start → delta* → end`
    /// keyed by id. Unbalanced triads make interleaved blocks impossible to
    /// reassemble downstream.
    private static func checkBalancedTriads(_ parts: [StreamPart], name: String) {
        checkTriad(
            parts, name: name, label: "text",
            start: { if case .textStart(let id, _) = $0 { return id } else { return nil } },
            delta: { if case .textDelta(let id, _, _) = $0 { return id } else { return nil } },
            end: { if case .textEnd(let id, _) = $0 { return id } else { return nil } }
        )
        checkTriad(
            parts, name: name, label: "reasoning",
            start: { if case .reasoningStart(let id, _) = $0 { return id } else { return nil } },
            delta: { if case .reasoningDelta(let id, _, _) = $0 { return id } else { return nil } },
            end: { if case .reasoningEnd(let id, _) = $0 { return id } else { return nil } }
        )
        checkTriad(
            parts, name: name, label: "tool input",
            start: { if case .toolInputStart(let id, _, _, _, _, _) = $0 { return id } else { return nil } },
            delta: { if case .toolInputDelta(let id, _, _) = $0 { return id } else { return nil } },
            end: { if case .toolInputEnd(let id, _) = $0 { return id } else { return nil } }
        )
    }

    private static func checkTriad(
        _ parts: [StreamPart],
        name: String,
        label: String,
        start: (StreamPart) -> String?,
        delta: (StreamPart) -> String?,
        end: (StreamPart) -> String?
    ) {
        var open: Set<String> = []

        for part in parts {
            if let id = start(part) {
                #expect(!open.contains(id), "\(name): \(label) \(id) started twice")
                open.insert(id)
            } else if let id = delta(part) {
                #expect(open.contains(id), "\(name): \(label) delta for unopened id \(id)")
            } else if let id = end(part) {
                #expect(open.contains(id), "\(name): \(label) \(id) ended without starting")
                open.remove(id)
            }
        }

        #expect(open.isEmpty, "\(name): unclosed \(label) blocks: \(open.sorted())")
    }

    private static func checkToolCalls(_ parts: [StreamPart], name: String) {
        var opened: Set<String> = []

        for part in parts {
            switch part {
            case .toolInputStart(let id, _, _, _, _, _):
                opened.insert(id)

            case .toolCall(let call):
                #expect(
                    opened.contains(call.toolCallId),
                    "\(name): tool call \(call.toolCallId) has no preceding tool-input-start"
                )
                // Arguments stream in as fragments that are individually
                // invalid JSON. Reassembly being off by one character produces
                // an unusable call, and the failure would otherwise surface
                // only at runtime against a live provider.
                #expect(
                    (try? JSONValue.decode(from: call.input)) != nil,
                    "\(name): tool call \(call.toolName) produced unparseable input: \(call.input)"
                )
                #expect(!call.toolName.isEmpty, "\(name): tool call \(call.toolCallId) has no name")

            default:
                break
            }
        }
    }

    private static func checkUsage(_ parts: [StreamPart], name: String) {
        for part in parts {
            guard case .finish(let usage, _, _) = part else { continue }

            let noCache = usage.inputTokens.noCache ?? 0
            let cacheRead = usage.inputTokens.cacheRead ?? 0
            let cacheWrite = usage.inputTokens.cacheWrite ?? 0

            #expect(noCache >= 0 && cacheRead >= 0 && cacheWrite >= 0, "\(name): negative token count")
            #expect((usage.outputTokens.total ?? 0) >= 0, "\(name): negative output tokens")

            // Providers disagree on whether the reported input total includes
            // cached tokens, so the identity is checked only when the mapper
            // claimed to know every leg.
            if let total = usage.inputTokens.total, usage.inputTokens.noCache != nil {
                #expect(
                    total == noCache + cacheRead + cacheWrite,
                    "\(name): input total \(total) != \(noCache) + \(cacheRead) + \(cacheWrite)"
                )
            }
        }
    }

    private static func checkNoParseErrors(_ parts: [StreamPart], name: String) {
        for part in parts {
            guard case .error(let error) = part else { continue }
            #expect(error.type != "parse_error", "\(name): chunk failed to decode: \(error.message)")
        }
    }
}
