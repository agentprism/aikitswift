import Foundation
import Testing

@testable import Manifold

/// Replays every recorded provider response through the mapper and asserts the
/// normalized stream is well-formed.
///
/// This is the test strategy the whole library rests on: the fixtures are real
/// captured API responses, so the mapper is exercised against real wire
/// behaviour — including compaction, context editing, server tools and
/// fallbacks — without a single API key.
///
/// A fixture file may hold several sequential API calls, so each is replayed
/// through its own mapper instance, exactly as production would.
@Suite("Anthropic wire conformance")
struct AnthropicConformanceTests {

    /// Chunk types the mapper understands. Anything else must surface as `raw`.
    static let knownChunkTypes: Set<String> = [
        "ping", "message_start", "message_delta", "message_stop",
        "content_block_start", "content_block_delta", "content_block_stop",
        "error",
    ]

    static var streamNames: [String] {
        get throws { try Fixture.anthropicStreamNames }
    }

    @Test("fixtures are present")
    func fixturesArePresent() throws {
        // Guards against a resource-bundling mistake quietly turning every
        // parameterized test below into a no-op.
        #expect(try Self.streamNames.count >= 20)
    }

    @Test("every message starts with stream-start", arguments: try streamNames)
    func startsWithStreamStart(name: String) throws {
        for (index, parts) in try Fixture.replayAnthropic(name).enumerated() {
            guard case .streamStart = parts.first else {
                Issue.record("\(name)[\(index)]: expected stream-start first, got \(String(describing: parts.first))")
                continue
            }
        }
    }

    @Test("every message reports response metadata", arguments: try streamNames)
    func reportsResponseMetadata(name: String) throws {
        for (index, parts) in try Fixture.replayAnthropic(name).enumerated() {
            let hasMetadata = parts.contains {
                if case .responseMetadata(let metadata) = $0 { return metadata.modelId != nil }
                return false
            }
            // The serving model is not always the requested one — a server-side
            // fallback can reroute a turn — so it must always be surfaced.
            #expect(hasMetadata, "\(name)[\(index)]: no response metadata carrying a model id")
        }
    }

    @Test("finish appears exactly once and last", arguments: try streamNames)
    func finishesExactlyOnce(name: String) throws {
        for (index, parts) in try Fixture.replayAnthropic(name).enumerated() {
            let finishIndices = parts.indices.filter {
                if case .finish = parts[$0] { return true }
                return false
            }

            #expect(
                finishIndices.count == 1,
                "\(name)[\(index)]: expected one finish, found \(finishIndices.count)"
            )
            #expect(
                finishIndices.last == parts.count - 1,
                "\(name)[\(index)]: finish is not the last part"
            )
        }
    }

    @Test("text blocks are balanced start → delta* → end", arguments: try streamNames)
    func balancesTextBlocks(name: String) throws {
        try assertBalanced(
            name: name,
            starts: { if case .textStart(let id, _) = $0 { return id } else { return nil } },
            deltas: { if case .textDelta(let id, _, _) = $0 { return id } else { return nil } },
            ends: { if case .textEnd(let id, _) = $0 { return id } else { return nil } },
            label: "text"
        )
    }

    @Test("reasoning blocks are balanced start → delta* → end", arguments: try streamNames)
    func balancesReasoningBlocks(name: String) throws {
        try assertBalanced(
            name: name,
            starts: { if case .reasoningStart(let id, _) = $0 { return id } else { return nil } },
            deltas: { if case .reasoningDelta(let id, _, _) = $0 { return id } else { return nil } },
            ends: { if case .reasoningEnd(let id, _) = $0 { return id } else { return nil } },
            label: "reasoning"
        )
    }

    @Test("tool input blocks are balanced start → delta* → end", arguments: try streamNames)
    func balancesToolInputBlocks(name: String) throws {
        try assertBalanced(
            name: name,
            starts: { if case .toolInputStart(let id, _, _, _, _, _) = $0 { return id } else { return nil } },
            deltas: { if case .toolInputDelta(let id, _, _) = $0 { return id } else { return nil } },
            ends: { if case .toolInputEnd(let id, _) = $0 { return id } else { return nil } },
            label: "tool input"
        )
    }

    @Test("assembled tool call arguments are valid JSON", arguments: try streamNames)
    func assemblesValidToolInput(name: String) throws {
        // Tool arguments stream in as fragments that are individually invalid
        // JSON. If reassembly is off by even one character the call is
        // unusable, and the failure would otherwise surface only at runtime
        // against a live provider.
        for parts in try Fixture.replayAnthropic(name) {
            for part in parts {
                guard case .toolCall(let call) = part else { continue }

                #expect(
                    (try? JSONValue.decode(from: call.input)) != nil,
                    "\(name): tool call \(call.toolName) produced unparseable input: \(call.input)"
                )
            }
        }
    }

    @Test("every tool call is preceded by its input block", arguments: try streamNames)
    func toolCallFollowsItsInputBlock(name: String) throws {
        for (index, parts) in try Fixture.replayAnthropic(name).enumerated() {
            var opened: Set<String> = []

            for part in parts {
                switch part {
                case .toolInputStart(let id, _, _, _, _, _):
                    opened.insert(id)
                case .toolCall(let call):
                    #expect(
                        opened.contains(call.toolCallId),
                        "\(name)[\(index)]: tool call \(call.toolCallId) has no preceding tool-input-start"
                    )
                default:
                    break
                }
            }
        }
    }

    @Test("reported usage is internally consistent", arguments: try streamNames)
    func reportsConsistentUsage(name: String) throws {
        for (index, parts) in try Fixture.replayAnthropic(name).enumerated() {
            for part in parts {
                guard case .finish(let usage, _, _) = part else { continue }

                let noCache = usage.inputTokens.noCache ?? 0
                let cacheRead = usage.inputTokens.cacheRead ?? 0
                let cacheWrite = usage.inputTokens.cacheWrite ?? 0

                #expect(
                    noCache >= 0 && cacheRead >= 0 && cacheWrite >= 0,
                    "\(name)[\(index)]: negative token count"
                )
                #expect(
                    usage.inputTokens.total == noCache + cacheRead + cacheWrite,
                    "\(name)[\(index)]: input total does not equal the sum of its legs"
                )
                #expect((usage.outputTokens.total ?? 0) >= 0, "\(name)[\(index)]: negative output tokens")
                #expect(usage.raw != nil, "\(name)[\(index)]: provider usage payload was dropped")
            }
        }
    }

    @Test("unrecognized chunks surface as raw", arguments: try streamNames)
    func surfacesUnknownChunksAsRaw(name: String) throws {
        // A chunk the mapper does not understand must never vanish. This is
        // what lets a provider ship a new event type without this library
        // silently losing data.
        for chunks in try Fixture.anthropicMessages(name) {
            var wire = AnthropicMessagesWire()

            for chunk in chunks {
                let type = chunk["type"]?.stringValue ?? "?"
                let parts = wire.map(chunk: chunk)

                guard !Self.knownChunkTypes.contains(type) else { continue }

                let hasRaw = parts.contains { if case .raw = $0 { return true } else { return false } }
                #expect(hasRaw, "\(name): unrecognized chunk type \(type) was dropped")
            }
        }
    }

    @Test("no chunk fails to decode", arguments: try streamNames)
    func decodesEveryChunk(name: String) throws {
        for parts in try Fixture.replayAnthropic(name) {
            for part in parts {
                guard case .error(let error) = part else { continue }
                #expect(error.type != "parse_error", "\(name): chunk failed to decode: \(error.message)")
            }
        }
    }

    // MARK: - Helper

    /// Asserts a start → delta* → end triad is well-formed for every id, in
    /// every message of the fixture.
    private func assertBalanced(
        name: String,
        starts: (StreamPart) -> String?,
        deltas: (StreamPart) -> String?,
        ends: (StreamPart) -> String?,
        label: String
    ) throws {
        for (index, parts) in try Fixture.replayAnthropic(name).enumerated() {
            var open: Set<String> = []

            for part in parts {
                if let id = starts(part) {
                    #expect(!open.contains(id), "\(name)[\(index)]: \(label) \(id) started twice")
                    open.insert(id)
                } else if let id = deltas(part) {
                    #expect(open.contains(id), "\(name)[\(index)]: \(label) delta for unopened id \(id)")
                } else if let id = ends(part) {
                    #expect(open.contains(id), "\(name)[\(index)]: \(label) \(id) ended without starting")
                    open.remove(id)
                }
            }

            #expect(open.isEmpty, "\(name)[\(index)]: unclosed \(label) blocks: \(open.sorted())")
        }
    }
}
