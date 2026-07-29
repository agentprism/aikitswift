import Foundation

/// Converts an Anthropic `usage` payload into normalized ``Usage``.
///
/// Isolated from the stream mapper because it is the subtlest arithmetic in the
/// whole wire layer: the top-level totals do *not* always describe what you
/// were billed for, and getting it wrong silently under-reports cost.
enum AnthropicUsageConverter {
    /// - Parameter usage: the provider's `usage` object.
    static func convert(_ usage: JSONValue) -> Usage {
        let cacheWrite = usage["cache_creation_input_tokens"]?.intValue ?? 0
        let cacheRead = usage["cache_read_input_tokens"]?.intValue ?? 0
        let reasoningTokens = usage["output_tokens_details"]?["thinking_tokens"]?.intValue

        let iterations = usage["iterations"]?.arrayValue ?? []

        // A turn served by a server-side fallback is the exception to the
        // summing rule below: the top-level totals already describe the answer
        // the fallback model produced, while the executor iteration is the
        // blocked primary attempt with zero output. Summing would erase the
        // real answer's cost.
        let servedByFallback = iterations.contains {
            $0["type"]?.stringValue == "fallback_message"
        }

        var inputTokens = usage["input_tokens"]?.intValue ?? 0
        var outputTokens = usage["output_tokens"]?.intValue ?? 0

        if !iterations.isEmpty && !servedByFallback {
            // With compaction or the advisor tool in play, the top-level
            // input/output counts exclude some iterations. Sum the executor
            // ones to recover the true totals.
            //
            // `advisor_message` iterations are deliberately excluded: they bill
            // at the advisor model's rates, not the executor's, so folding them
            // into one total would misprice both.
            let executor = iterations.filter {
                let type = $0["type"]?.stringValue
                return type == "compaction" || type == "message"
            }

            if !executor.isEmpty {
                inputTokens = executor.reduce(0) { $0 + ($1["input_tokens"]?.intValue ?? 0) }
                outputTokens = executor.reduce(0) { $0 + ($1["output_tokens"]?.intValue ?? 0) }
            }
        }

        return Usage(
            inputTokens: .init(
                // `input_tokens` from the API counts only uncached tokens, so
                // the true total has to add the cache legs back in.
                total: inputTokens + cacheWrite + cacheRead,
                noCache: inputTokens,
                cacheRead: cacheRead,
                cacheWrite: cacheWrite
            ),
            outputTokens: .init(
                total: outputTokens,
                text: reasoningTokens.map { outputTokens - $0 },
                reasoning: reasoningTokens
            ),
            raw: usage
        )
    }
}
