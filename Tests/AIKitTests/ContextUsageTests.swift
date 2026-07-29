import Foundation
import Testing

@testable import AIKit

@Suite("Token counting and context attribution")
struct ContextUsageTests {

    @Test("CJK text costs far more per character than Latin")
    func weighsCJKCorrectly() {
        let tokenizer = HeuristicTokenizer()

        // A uniform characters/4 rule under-counts Chinese roughly fourfold —
        // the difference between "half full" and "over the limit".
        let latin = tokenizer.count(String(repeating: "a", count: 100))
        let chinese = tokenizer.count(String(repeating: "字", count: 100))

        #expect(latin < 40)
        #expect(chinese >= 90)
        #expect(chinese > latin * 2)
    }

    @Test("mixed scripts are counted per script")
    func handlesMixedScripts() {
        let tokenizer = HeuristicTokenizer()
        let mixed = tokenizer.count("hello 世界")

        #expect(mixed >= 3)
        #expect(mixed <= 8)
    }

    @Test("empty text is free, any text costs at least one token")
    func handlesEdges() {
        let tokenizer = HeuristicTokenizer()

        #expect(tokenizer.count("") == 0)
        #expect(tokenizer.count("a") == 1)
    }

    @Test("segments are attributed to their category")
    func attributesSegments() {
        let options = CallOptions(
            model: "test",
            prompt: [
                .system(String(repeating: "system instructions ", count: 50)),
                .user(String(repeating: "user question ", count: 10)),
            ],
            tools: [ToolDefinition(
                name: "search",
                description: "Search the web",
                inputSchema: ["type": "object", "properties": ["q": ["type": "string"]]]
            )]
        )

        let usage = ContextReporter().report(options, contextWindow: 200_000)
        let segments = usage.entries.map(\.segment)

        #expect(segments.contains(.systemPrompt))
        #expect(segments.contains(.messages))
        #expect(segments.contains(.tools))

        // The system prompt is five times the user turn here, and attribution
        // that cannot tell them apart is useless for deciding what to trim.
        let system = usage.entries.first { $0.segment == .systemPrompt }?.tokens ?? 0
        let messages = usage.entries.first { $0.segment == .messages }?.tokens ?? 0
        #expect(system > messages)
    }

    @Test("tool schemas are counted, not just descriptions")
    func countsToolSchemas() {
        // A tool surface is often the quietest consumer of a context window,
        // and the schema is usually larger than the prose.
        let bigSchema = ToolDefinition(
            name: "x",
            description: "short",
            inputSchema: .object(Dictionary(
                uniqueKeysWithValues: (0..<40).map {
                    ("field_number_\($0)", JSONValue.object(["type": "string", "description": "a field"]))
                }
            ))
        )

        let usage = ContextReporter().report(
            CallOptions(model: "test", prompt: [.user("hi")], tools: [bigSchema])
        )
        let tools = usage.entries.first { $0.segment == .tools }?.tokens ?? 0

        #expect(tools > 100)
    }

    @Test("free space and utilization derive from the context window")
    func computesFreeSpace() {
        let usage = ContextUsage(
            entries: [.init(segment: .messages, tokens: 50_000)],
            contextWindow: 200_000
        )

        #expect(usage.used == 50_000)
        #expect(usage.freeSpace == 150_000)
        #expect(usage.utilization == 0.25)
    }

    @Test("over-full context reports no free space rather than a negative")
    func clampsFreeSpace() {
        // A request can exceed the window. Negative "remaining" reads as a bug
        // in whatever renders it.
        let usage = ContextUsage(
            entries: [.init(segment: .messages, tokens: 300_000)],
            contextWindow: 200_000
        )

        #expect(usage.freeSpace == 0)
        #expect((usage.utilization ?? 0) > 1)
    }

    @Test("calibration makes the total exact and keeps proportions")
    func calibratesToExactTotal() {
        // The exact total is free: it is the previous response's input usage.
        // Anchoring to it beats paying for per-segment precision.
        let estimated = ContextUsage(
            entries: [
                .init(segment: .systemPrompt, tokens: 100),
                .init(segment: .messages, tokens: 300),
            ],
            contextWindow: 10_000
        )

        let exact = estimated.calibrated(toTotal: 800)

        #expect(exact.used == 800)
        #expect(exact.isEstimated == false)

        // Proportions survive: 1:3 before, 1:3 after.
        let system = exact.entries.first { $0.segment == .systemPrompt }?.tokens ?? 0
        let messages = exact.entries.first { $0.segment == .messages }?.tokens ?? 0
        #expect(system == 200)
        #expect(messages == 600)
    }

    @Test("calibration absorbs rounding drift so the parts sum to the whole")
    func absorbsRoundingDrift() {
        // Rounding each segment independently leaves the sum a few tokens off.
        let estimated = ContextUsage(entries: [
            .init(segment: .systemPrompt, tokens: 1),
            .init(segment: .messages, tokens: 1),
            .init(segment: .tools, tokens: 1),
        ])

        let exact = estimated.calibrated(toTotal: 100)
        #expect(exact.used == 100)
    }

    @Test("calibration is safe on an empty estimate")
    func toleratesEmptyEstimate() {
        let empty = ContextUsage(entries: [])
        #expect(empty.calibrated(toTotal: 500).isEstimated == false)
    }

    @Test("token counts render the way a context readout does")
    func formatsTokenCounts() {
        #expect(ContextUsage.format(284) == "284")
        #expect(ContextUsage.format(499_600) == "499.6k")
        #expect(ContextUsage.format(1_000_000) == "1.0M")
    }

    @Test("context window can be read from the catalog")
    func readsWindowFromCatalog() {
        // Wiring the real denominator should not require hardcoding a number
        // that changes with every model release.
        guard let (_, model) = ProviderCatalog.model("claude-opus-4-8", provider: "anthropic") else {
            Issue.record("expected claude-opus-4-8 in the catalog")
            return
        }

        let usage = ContextReporter().report(
            CallOptions(model: model.id, prompt: [.user("hi")]),
            contextWindow: model.contextWindow
        )

        #expect((usage.contextWindow ?? 0) > 0)
        #expect((usage.freeSpace ?? 0) > 0)
    }
}
