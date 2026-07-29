import Foundation
import Testing

@testable import Manifold

/// One recorded stream, identified by which fixture set it came from.
struct FixtureRef: Sendable, CustomStringConvertible {
    let set: String
    let name: String

    var description: String { "\(set)/\(name)" }
}

/// Replays recordings from seven different vendors through the *same* mapper.
///
/// This is the payoff of splitting by protocol rather than by vendor: DeepSeek,
/// xAI, Groq, Mistral, Cerebras and the generic OpenAI-compatible endpoints all
/// speak Chat Completions, so one implementation is validated against all of
/// their real traffic at once. Thirty-eight of the fifty providers in the
/// catalog route through this code path.
@Suite("OpenAI Completions wire")
struct OpenAICompletionsConformanceTests {

    /// Fixture sets whose recordings speak Chat Completions.
    static let sets = [
        "openai-completions", "deepseek", "xai", "groq",
        "mistral", "cerebras", "openai-compatible",
    ]

    static var recordings: [FixtureRef] {
        get throws {
            try sets.flatMap { set in
                ((try? Fixture.streamNames(set)) ?? []).map { FixtureRef(set: set, name: $0) }
            }
        }
    }

    @Test("fixtures span multiple vendors")
    func spansMultipleVendors() throws {
        let sets = Set(try Self.recordings.map(\.set))
        // The point of this suite is cross-vendor coverage; if the sync script
        // only landed one vendor, the suite is not testing what it claims to.
        #expect(sets.count >= 5, "expected recordings from at least 5 vendors, got \(sets.sorted())")
        #expect(try Self.recordings.count >= 15)
    }

    @Test("recorded streams are well-formed", arguments: try recordings)
    func conforms(ref: FixtureRef) throws {
        WireConformance.check(
            try Fixture.replay(OpenAICompletionsWire.self, ref.set, ref.name),
            label: ref.description
        )
    }

    @Test("finish is idempotent", arguments: try recordings)
    func finishIsIdempotent(ref: FixtureRef) throws {
        // The client calls `finish()` on connection close, and `[DONE]` may
        // also have triggered it. Emitting a second terminal `finish` would
        // break every downstream consumer that stops on the first.
        var wire = OpenAICompletionsWire()
        for chunk in try Fixture.chunks(ref.set, ref.name) {
            _ = wire.map(chunk: chunk)
        }

        let first = wire.finish()
        let second = wire.finish()

        #expect(first.contains { if case .finish = $0 { return true } else { return false } })
        #expect(second.isEmpty, "\(ref): finish() emitted parts on the second call")
    }

    @Test("finish reasons map to the normalized set")
    func mapsFinishReasons() {
        #expect(OpenAICompletionsWire.mapFinishReason("stop").unified == .stop)
        #expect(OpenAICompletionsWire.mapFinishReason("length").unified == .length)
        #expect(OpenAICompletionsWire.mapFinishReason("tool_calls").unified == .toolCalls)
        #expect(OpenAICompletionsWire.mapFinishReason("function_call").unified == .toolCalls)
        #expect(OpenAICompletionsWire.mapFinishReason("content_filter").unified == .contentFilter)

        let future = OpenAICompletionsWire.mapFinishReason("something_new")
        #expect(future.unified == .other)
        #expect(future.raw == "something_new")
    }

    @Test("cached tokens are subtracted, not added")
    func treatsCachedTokensAsIncluded() {
        // OpenAI's `prompt_tokens` *includes* cached tokens; Anthropic's
        // `input_tokens` excludes them. Applying Anthropic's arithmetic here
        // would double-count the cache — an error that never raises and only
        // shows up on a bill.
        let usage = OpenAICompletionsWire.convertUsage([
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "prompt_tokens_details": ["cached_tokens": 800],
        ])

        #expect(usage.inputTokens.total == 1000)
        #expect(usage.inputTokens.noCache == 200)
        #expect(usage.inputTokens.cacheRead == 800)
    }

    @Test("reasoning tokens are split out of the completion total")
    func splitsReasoningTokens() {
        let usage = OpenAICompletionsWire.convertUsage([
            "prompt_tokens": 10,
            "completion_tokens": 500,
            "completion_tokens_details": ["reasoning_tokens": 400],
        ])

        #expect(usage.outputTokens.total == 500)
        #expect(usage.outputTokens.reasoning == 400)
        #expect(usage.outputTokens.text == 100)
    }

    @Test("reasoning is accepted under all three spellings")
    func acceptsAllReasoningSpellings() {
        // DeepSeek uses `reasoning_content`; others use `reasoning`, as a bare
        // string or as an object. Rejecting any one of them would turn a
        // vendor quirk into a missing feature.
        for delta in [
            JSONValue.object(["reasoning_content": .string("thinking")]),
            JSONValue.object(["reasoning": .string("thinking")]),
            JSONValue.object(["reasoning": .object(["text": .string("thinking")])]),
        ] {
            var wire = OpenAICompletionsWire()
            let parts = wire.map(chunk: [
                "id": "x",
                "choices": [["index": 0, "delta": delta]],
            ])

            let hasReasoning = parts.contains {
                if case .reasoningDelta(_, let text, _) = $0 { return text == "thinking" }
                return false
            }
            #expect(hasReasoning, "reasoning not recognized in \(delta)")
        }
    }

    @Test("tool call ids survive being omitted after the first chunk")
    func assemblesToolCallAcrossChunks() {
        // Only the first chunk carries the id and name; later fragments
        // identify the call by array index alone.
        var wire = OpenAICompletionsWire()

        _ = wire.map(chunk: [
            "id": "x",
            "choices": [["index": 0, "delta": ["tool_calls": [
                ["index": 0, "id": "call_abc", "function": ["name": "get_weather", "arguments": ""]]
            ]]]],
        ])
        _ = wire.map(chunk: [
            "choices": [["index": 0, "delta": ["tool_calls": [
                ["index": 0, "function": ["arguments": "{\"city\":"]]
            ]]]],
        ])
        _ = wire.map(chunk: [
            "choices": [["index": 0, "delta": ["tool_calls": [
                ["index": 0, "function": ["arguments": "\"Paris\"}"]]
            ]]]],
        ])

        let parts = wire.finish()
        let call = parts.compactMap { if case .toolCall(let c) = $0 { return c } else { return nil } }.first

        #expect(call?.toolCallId == "call_abc")
        #expect(call?.toolName == "get_weather")
        #expect(call?.input == "{\"city\":\"Paris\"}")
    }
}
