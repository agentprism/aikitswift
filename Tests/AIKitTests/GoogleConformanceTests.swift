import Foundation
import Testing

@testable import AIKit

@Suite("Google Generative AI wire")
struct GoogleConformanceTests {

    static let set = "google"

    static var streamNames: [String] {
        get throws { try Fixture.streamNames(set) }
    }

    @Test("fixtures are present")
    func fixturesArePresent() throws {
        #expect(try Self.streamNames.count >= 15)
    }

    @Test("recorded streams are well-formed", arguments: try streamNames)
    func conforms(name: String) throws {
        WireConformance.check(
            try Fixture.replay(GoogleGenerativeAIWire.self, Self.set, name),
            label: name
        )
    }

    @Test("finish reasons map to the normalized set")
    func mapsFinishReasons() {
        #expect(GoogleGenerativeAIWire.mapFinishReason("STOP").unified == .stop)
        #expect(GoogleGenerativeAIWire.mapFinishReason("MAX_TOKENS").unified == .length)
        #expect(GoogleGenerativeAIWire.mapFinishReason("SAFETY").unified == .contentFilter)
        #expect(GoogleGenerativeAIWire.mapFinishReason("RECITATION").unified == .contentFilter)
        #expect(GoogleGenerativeAIWire.mapFinishReason("MALFORMED_FUNCTION_CALL").unified == .error)

        let future = GoogleGenerativeAIWire.mapFinishReason("SOMETHING_NEW")
        #expect(future.unified == .other)
        #expect(future.raw == "SOMETHING_NEW")
    }

    @Test("thinking tokens are added to the output total, not subtracted")
    func addsThoughtTokensToOutput() {
        // Gemini's `candidatesTokenCount` excludes thinking tokens — the
        // opposite of OpenAI. The provider's own arithmetic confirms it:
        // totalTokenCount = prompt + candidates + thoughts.
        let usage = GoogleGenerativeAIWire.convertUsage([
            "promptTokenCount": 29,
            "candidatesTokenCount": 15,
            "thoughtsTokenCount": 804,
            "totalTokenCount": 848,
        ])

        #expect(usage.inputTokens.total == 29)
        #expect(usage.outputTokens.text == 15)
        #expect(usage.outputTokens.reasoning == 804)
        #expect(usage.outputTokens.total == 819)

        // Round-trips to the provider's own total.
        let total = (usage.inputTokens.total ?? 0) + (usage.outputTokens.total ?? 0)
        #expect(total == 848)
    }

    @Test("cached content is split out of the prompt total")
    func splitsCachedContent() {
        let usage = GoogleGenerativeAIWire.convertUsage([
            "promptTokenCount": 1000,
            "candidatesTokenCount": 50,
            "cachedContentTokenCount": 700,
        ])

        #expect(usage.inputTokens.total == 1000)
        #expect(usage.inputTokens.noCache == 300)
        #expect(usage.inputTokens.cacheRead == 700)
    }

    @Test("a whole-args function call still produces a full input triad")
    func emitsFullTriadForWholeArgs() {
        // Gemini delivers complete arguments in one part. Downstream consumers
        // rely on the triad regardless, so it is emitted in one step rather
        // than special-cased.
        var wire = GoogleGenerativeAIWire()
        let parts = wire.map(chunk: [
            "candidates": [[
                "index": 0,
                "content": ["parts": [[
                    "functionCall": ["name": "weather", "args": ["location": "San Francisco"]]
                ]]],
            ]],
        ])

        #expect(parts.contains { if case .toolInputStart = $0 { return true } else { return false } })
        #expect(parts.contains { if case .toolInputDelta = $0 { return true } else { return false } })
        #expect(parts.contains { if case .toolInputEnd = $0 { return true } else { return false } })

        let call = parts.compactMap { if case .toolCall(let c) = $0 { return c } else { return nil } }.first
        #expect(call?.toolName == "weather")
        #expect(call?.input == #"{"location":"San Francisco"}"#)
    }

    @Test("synthesized tool call ids are deterministic")
    func synthesizesDeterministicIds() {
        // Gemini assigns no call id. Replaying the same stream must produce the
        // same ids, or nothing downstream can be compared or cached.
        func run() -> [String] {
            var wire = GoogleGenerativeAIWire()
            let parts = wire.map(chunk: [
                "candidates": [[
                    "index": 0,
                    "content": ["parts": [
                        ["functionCall": ["name": "a", "args": [:]]],
                        ["functionCall": ["name": "b", "args": [:]]],
                    ]],
                ]],
            ])
            return parts.compactMap { if case .toolCall(let c) = $0 { return c.toolCallId } else { return nil } }
        }

        #expect(run() == ["a-1", "b-2"])
        #expect(run() == run())
    }

    @Test("response id and serving model may arrive on different chunks")
    func accumulatesResponseMetadata() {
        var wire = GoogleGenerativeAIWire()

        let first = wire.map(chunk: ["responseId": "response-1"])
        let second = wire.map(chunk: ["modelVersion": "gemini-fallback"])

        let firstMetadata = first.compactMap {
            if case .responseMetadata(let metadata) = $0 { metadata } else { nil }
        }.first
        let secondMetadata = second.compactMap {
            if case .responseMetadata(let metadata) = $0 { metadata } else { nil }
        }.first

        #expect(firstMetadata?.id == "response-1")
        #expect(firstMetadata?.modelId == nil)
        #expect(secondMetadata?.id == "response-1")
        #expect(secondMetadata?.modelId == "gemini-fallback")
    }

    @Test("a thought part closes an open text block")
    func switchesBetweenTextAndReasoning() {
        // Reasoning is a flag on a text part, not a distinct block type, so the
        // transition has to be inferred or the triads go unbalanced.
        var wire = GoogleGenerativeAIWire()

        _ = wire.map(chunk: ["candidates": [[
            "index": 0, "content": ["parts": [["text": "answer"]]],
        ]]])
        let parts = wire.map(chunk: ["candidates": [[
            "index": 0, "content": ["parts": [["text": "hmm", "thought": true]]],
        ]]])

        #expect(parts.contains { if case .textEnd = $0 { return true } else { return false } })
        #expect(parts.contains { if case .reasoningStart = $0 { return true } else { return false } })
    }
}
