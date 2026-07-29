import Foundation
import Testing

@testable import AIKit

@Suite("OpenAI Responses wire")
struct OpenAIResponsesConformanceTests {

    static let set = "openai-responses"

    static var streamNames: [String] {
        get throws { try Fixture.streamNames(set) }
    }

    @Test("fixtures are present")
    func fixturesArePresent() throws {
        #expect(try Self.streamNames.count >= 20)
    }

    @Test("recorded streams are well-formed", arguments: try streamNames)
    func conforms(name: String) throws {
        WireConformance.check(
            try Fixture.replay(OpenAIResponsesWire.self, Self.set, name),
            label: name
        )
    }

    @Test("statuses map to the normalized set")
    func mapsStatuses() {
        #expect(OpenAIResponsesWire.mapStatus("completed").unified == .stop)
        // The API reports success without saying tools are pending, so the
        // presence of a call is the only signal the caller gets.
        #expect(OpenAIResponsesWire.mapStatus("completed", hasToolCalls: true).unified == .toolCalls)
        #expect(
            OpenAIResponsesWire.mapStatus("incomplete", incompleteReason: "max_output_tokens").unified == .length
        )
        #expect(
            OpenAIResponsesWire.mapStatus("incomplete", incompleteReason: "content_filter").unified == .contentFilter
        )
        #expect(OpenAIResponsesWire.mapStatus("failed").unified == .error)

        let future = OpenAIResponsesWire.mapStatus("something_new")
        #expect(future.unified == .other)
        #expect(future.raw == "something_new")
    }

    @Test("cached and reasoning tokens are subtracted, not added")
    func treatsDetailsAsIncluded() {
        // Both totals are inclusive here, matching Chat Completions and
        // differing from Anthropic (input) and Gemini (output).
        let usage = OpenAIResponsesWire.convertUsage([
            "input_tokens": 1000,
            "output_tokens": 500,
            "input_tokens_details": ["cached_tokens": 900],
            "output_tokens_details": ["reasoning_tokens": 400],
        ])

        #expect(usage.inputTokens.total == 1000)
        #expect(usage.inputTokens.noCache == 100)
        #expect(usage.inputTokens.cacheRead == 900)
        #expect(usage.outputTokens.total == 500)
        #expect(usage.outputTokens.text == 100)
        #expect(usage.outputTokens.reasoning == 400)
    }

    @Test("tool results reference call_id, not the streaming item id")
    func usesCallIdForToolCalls() {
        // `id` addresses the streaming item; `call_id` is what a tool result
        // must reference. Confusing them produces a result the model cannot
        // match back to its call.
        var wire = OpenAIResponsesWire()

        _ = wire.map(chunk: [
            "type": "response.output_item.added",
            "item": ["type": "function_call", "id": "item_1", "call_id": "call_xyz", "name": "get_weather"],
        ])
        _ = wire.map(chunk: [
            "type": "response.function_call_arguments.delta",
            "item_id": "item_1",
            "delta": "{\"city\":\"Paris\"}",
        ])
        let parts = wire.map(chunk: [
            "type": "response.output_item.done",
            "item": ["type": "function_call", "id": "item_1"],
        ])

        let call = parts.compactMap { if case .toolCall(let c) = $0 { return c } else { return nil } }.first
        #expect(call?.toolCallId == "call_xyz")
        #expect(call?.toolName == "get_weather")
        #expect(call?.input == "{\"city\":\"Paris\"}")
    }

    @Test("text blocks are keyed by item and content index")
    func keysTextByItemAndContentIndex() {
        // A single item can carry several content parts. Keying on item alone
        // would collapse them and unbalance the triads.
        var wire = OpenAIResponsesWire()

        for index in 0..<2 {
            _ = wire.map(chunk: [
                "type": "response.content_part.added",
                "item_id": "item_1",
                "content_index": .number(Double(index)),
                "part": ["type": "output_text"],
            ])
        }

        let starts = wire.map(chunk: [
            "type": "response.content_part.done",
            "item_id": "item_1",
            "content_index": 0,
            "part": ["type": "output_text"],
        ])

        #expect(starts.contains {
            if case .textEnd(let id, _) = $0 { return id == "item_1#0" } else { return false }
        })
    }
}
