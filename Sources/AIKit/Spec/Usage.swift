import Foundation

/// Token usage for a single model call.
///
/// Every field is optional because providers disagree about what they report.
/// A `nil` means "this provider did not tell us", which is distinct from zero.
public struct Usage: Sendable, Hashable {
    public struct Input: Sendable, Hashable {
        /// Total input tokens, cached and uncached combined.
        public var total: Int?
        /// Input tokens that were *not* served from cache.
        public var noCache: Int?
        /// Input tokens served from cache (billed at a large discount).
        public var cacheRead: Int?
        /// Input tokens written to cache (billed at a premium).
        public var cacheWrite: Int?

        public init(total: Int? = nil, noCache: Int? = nil, cacheRead: Int? = nil, cacheWrite: Int? = nil) {
            self.total = total
            self.noCache = noCache
            self.cacheRead = cacheRead
            self.cacheWrite = cacheWrite
        }
    }

    public struct Output: Sendable, Hashable {
        /// Total output tokens, including reasoning.
        public var total: Int?
        /// Output tokens spent on visible text.
        public var text: Int?
        /// Output tokens spent on reasoning. Billed, whether or not it is shown.
        public var reasoning: Int?

        public init(total: Int? = nil, text: Int? = nil, reasoning: Int? = nil) {
            self.total = total
            self.text = text
            self.reasoning = reasoning
        }
    }

    public var inputTokens: Input
    public var outputTokens: Output
    /// The provider's own usage payload, unmodified.
    ///
    /// Providers report cost-relevant detail that the normalized shape has no
    /// room for — per-iteration breakdowns, service tiers, inference region.
    /// Anything you cannot get from the fields above is in here.
    public var raw: JSONValue?

    public init(inputTokens: Input = .init(), outputTokens: Output = .init(), raw: JSONValue? = nil) {
        self.inputTokens = inputTokens
        self.outputTokens = outputTokens
        self.raw = raw
    }

    public static let empty = Usage()
}
