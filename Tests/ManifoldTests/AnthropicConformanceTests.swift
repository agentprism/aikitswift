import Foundation
import Testing

@testable import Manifold

/// Replays every recorded Anthropic response through the mapper.
///
/// The fixtures are real captured API responses, so the mapper is exercised
/// against real wire behaviour — compaction, context editing, server tools,
/// fallbacks, reasoning signatures — without a single API key.
@Suite("Anthropic Messages wire")
struct AnthropicConformanceTests {

    static let set = "anthropic"

    /// Chunk types the mapper understands. Anything else must surface as `raw`.
    static let knownChunkTypes: Set<String> = [
        "ping", "message_start", "message_delta", "message_stop",
        "content_block_start", "content_block_delta", "content_block_stop",
        "error",
    ]

    static var streamNames: [String] {
        get throws { try Fixture.streamNames(set) }
    }

    @Test("fixtures are present")
    func fixturesArePresent() throws {
        // Guards against a resource-bundling mistake quietly turning every
        // parameterized test below into a no-op.
        #expect(try Self.streamNames.count >= 20)
    }

    @Test("recorded streams are well-formed", arguments: try streamNames)
    func conforms(name: String) throws {
        WireConformance.check(
            try Fixture.replay(
                AnthropicMessagesWire.self, Self.set, name,
                splitOn: Fixture.anthropicBoundary
            ),
            label: name
        )
    }

    @Test("every message reports the serving model", arguments: try streamNames)
    func reportsServingModel(name: String) throws {
        // A server-side fallback can reroute a turn to a different model, so
        // the model that answered is not always the one that was requested.
        for (index, parts) in try Fixture.replay(
            AnthropicMessagesWire.self, Self.set, name, splitOn: Fixture.anthropicBoundary
        ).enumerated() {
            let reported = parts.contains {
                if case .responseMetadata(let metadata) = $0 { return metadata.modelId != nil }
                return false
            }
            #expect(reported, "\(name)[\(index)]: no response metadata carrying a model id")
        }
    }

    @Test("unrecognized chunks surface as raw", arguments: try streamNames)
    func surfacesUnknownChunksAsRaw(name: String) throws {
        // A chunk the mapper does not understand must never vanish. This is
        // what lets a provider ship a new event type without this library
        // silently losing data.
        for chunks in try Fixture.streams(Self.set, name, splitOn: Fixture.anthropicBoundary) {
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

    @Test("stop reasons map to the normalized set")
    func mapsStopReasons() {
        #expect(AnthropicMessagesWire.mapStopReason("end_turn").unified == .stop)
        #expect(AnthropicMessagesWire.mapStopReason("stop_sequence").unified == .stop)
        #expect(AnthropicMessagesWire.mapStopReason("pause_turn").unified == .stop)
        #expect(AnthropicMessagesWire.mapStopReason("tool_use").unified == .toolCalls)
        #expect(AnthropicMessagesWire.mapStopReason("max_tokens").unified == .length)
        #expect(AnthropicMessagesWire.mapStopReason("model_context_window_exceeded").unified == .length)
        // A safety classifier declining is a content outcome, not an error:
        // the response is a normal 200 with empty or partial content.
        #expect(AnthropicMessagesWire.mapStopReason("refusal").unified == .contentFilter)

        // Unknown reasons degrade rather than crash, but keep the original.
        let future = AnthropicMessagesWire.mapStopReason("something_new")
        #expect(future.unified == .other)
        #expect(future.raw == "something_new")
    }
}
