import Testing

@testable import AIKit

/// Usage arithmetic is the subtlest part of the wire layer: the provider's
/// top-level totals do not always describe what you were billed for, and every
/// mistake here under-reports cost silently rather than failing loudly.
@Suite("Anthropic usage conversion")
struct AnthropicUsageTests {

    @Test("cache legs are added back into the input total")
    func addsCacheLegsToTotal() {
        // `input_tokens` from the API counts only uncached tokens.
        let usage = AnthropicUsageConverter.convert([
            "input_tokens": 100,
            "output_tokens": 50,
            "cache_creation_input_tokens": 20,
            "cache_read_input_tokens": 30,
        ])

        #expect(usage.inputTokens.total == 150)
        #expect(usage.inputTokens.noCache == 100)
        #expect(usage.inputTokens.cacheWrite == 20)
        #expect(usage.inputTokens.cacheRead == 30)
        #expect(usage.outputTokens.total == 50)
    }

    @Test("missing cache fields are treated as zero, not unknown")
    func defaultsCacheFieldsToZero() {
        let usage = AnthropicUsageConverter.convert([
            "input_tokens": 10,
            "output_tokens": 5,
        ])

        #expect(usage.inputTokens.total == 10)
        #expect(usage.inputTokens.cacheRead == 0)
        #expect(usage.inputTokens.cacheWrite == 0)
    }

    @Test("reasoning tokens are split out of the output total")
    func splitsReasoningTokens() {
        let usage = AnthropicUsageConverter.convert([
            "input_tokens": 10,
            "output_tokens": 100,
            "output_tokens_details": ["thinking_tokens": 40],
        ])

        #expect(usage.outputTokens.total == 100)
        #expect(usage.outputTokens.reasoning == 40)
        #expect(usage.outputTokens.text == 60)
    }

    @Test("text tokens stay unknown when the provider reports no reasoning split")
    func leavesTextNilWithoutReasoning() {
        let usage = AnthropicUsageConverter.convert([
            "input_tokens": 10,
            "output_tokens": 100,
        ])

        // nil means "not reported", which is not the same as zero.
        #expect(usage.outputTokens.text == nil)
        #expect(usage.outputTokens.reasoning == nil)
    }

    @Test("executor iterations are summed when compaction ran")
    func sumsExecutorIterations() {
        // With compaction in play the top-level totals exclude the compaction
        // pass, so they understate the real spend.
        let usage = AnthropicUsageConverter.convert([
            "input_tokens": 500,
            "output_tokens": 50,
            "iterations": [
                ["type": "compaction", "input_tokens": 1000, "output_tokens": 200],
                ["type": "message", "input_tokens": 500, "output_tokens": 50],
            ],
        ])

        #expect(usage.inputTokens.noCache == 1500)
        #expect(usage.outputTokens.total == 250)
    }

    @Test("advisor iterations are excluded from executor totals")
    func excludesAdvisorIterations() {
        // The advisor sub-inference bills at a different model's rates, so
        // folding it into one total would misprice both.
        let usage = AnthropicUsageConverter.convert([
            "input_tokens": 100,
            "output_tokens": 20,
            "iterations": [
                ["type": "message", "input_tokens": 100, "output_tokens": 20],
                ["type": "advisor_message", "input_tokens": 900, "output_tokens": 400],
            ],
        ])

        #expect(usage.inputTokens.noCache == 100)
        #expect(usage.outputTokens.total == 20)
    }

    @Test("a fallback-served turn uses the top-level totals verbatim")
    func usesTopLevelWhenServedByFallback() {
        // The executor iteration here is the blocked primary attempt with zero
        // output; the top-level totals already describe the fallback's answer.
        // Summing would erase the real answer's cost.
        let usage = AnthropicUsageConverter.convert([
            "input_tokens": 300,
            "output_tokens": 80,
            "iterations": [
                ["type": "message", "input_tokens": 300, "output_tokens": 0],
                ["type": "fallback_message", "input_tokens": 300, "output_tokens": 80],
            ],
        ])

        #expect(usage.inputTokens.noCache == 300)
        #expect(usage.outputTokens.total == 80)
    }

    @Test("the provider payload is preserved verbatim")
    func preservesRawPayload() {
        // Providers report cost-relevant detail the normalized shape has no
        // room for. Losing it means losing the ability to audit a bill.
        let raw: JSONValue = [
            "input_tokens": 10,
            "output_tokens": 5,
            "service_tier": "standard",
        ]

        #expect(AnthropicUsageConverter.convert(raw).raw == raw)
    }
}
