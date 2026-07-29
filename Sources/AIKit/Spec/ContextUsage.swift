import Foundation

/// Where a request's tokens went, broken down by category.
///
/// The shape behind a context-window readout: how much of the window each part
/// of the request occupies, and how much is left.
///
/// Segment counts are normally *estimated* while the total can be *exact* —
/// see ``calibrated(toTotal:)``. That combination is deliberate: a breakdown
/// needs correct proportions and an honest total, and exact per-segment counts
/// would cost one provider round trip per segment to buy precision nobody reads.
public struct ContextUsage: Sendable, Hashable {

    /// A category of request content.
    public enum Segment: Sendable, Hashable {
        /// System instructions.
        case systemPrompt
        /// Conversation history.
        case messages
        /// Tool definitions — schemas included, which is usually the surprise.
        case tools
        /// Anything the host application wants to account for separately:
        /// skills, memory files, retrieved documents.
        case custom(String)

        public var label: String {
            switch self {
            case .systemPrompt: "System prompt"
            case .messages: "Messages"
            case .tools: "Tools"
            case .custom(let name): name
            }
        }
    }

    public struct Entry: Sendable, Hashable {
        public var segment: Segment
        public var tokens: Int

        public init(segment: Segment, tokens: Int) {
            self.segment = segment
            self.tokens = tokens
        }
    }

    public var entries: [Entry]
    /// The model's context window, when known. The denominator for shares.
    public var contextWindow: Int?
    /// Whether the total is an estimate. `false` after calibration.
    public var isEstimated: Bool

    public init(entries: [Entry], contextWindow: Int? = nil, isEstimated: Bool = true) {
        self.entries = entries
        self.contextWindow = contextWindow
        self.isEstimated = isEstimated
    }

    /// Total tokens accounted for.
    public var used: Int {
        entries.reduce(0) { $0 + $1.tokens }
    }

    /// Tokens still available, when the window is known.
    ///
    /// Clamped at zero: a request can exceed the window, and a negative
    /// "remaining" reads as a bug in whatever renders it.
    public var freeSpace: Int? {
        contextWindow.map { max(0, $0 - used) }
    }

    /// Fraction of the context window in use, from 0 to 1.
    public var utilization: Double? {
        guard let contextWindow, contextWindow > 0 else { return nil }
        return Double(used) / Double(contextWindow)
    }

    /// One entry's share of the context window, or of the used total when the
    /// window is unknown.
    public func share(of entry: Entry) -> Double {
        let denominator = contextWindow ?? used
        guard denominator > 0 else { return 0 }
        return Double(entry.tokens) / Double(denominator)
    }

    /// Rescales every segment so the total matches a known-exact figure.
    ///
    /// The exact total is available for free: ``Usage/inputTokens`` on the
    /// previous response is authoritative and already paid for. Anchoring to it
    /// yields an exact total with proportionally-correct segments and no extra
    /// network call — which is why per-segment exactness is not worth chasing.
    ///
    /// Use a provider's token-counting endpoint instead only when the figure is
    /// needed *before* sending.
    public func calibrated(toTotal actual: Int) -> ContextUsage {
        let estimated = used
        guard estimated > 0, actual > 0 else {
            return ContextUsage(entries: entries, contextWindow: contextWindow, isEstimated: false)
        }

        let scale = Double(actual) / Double(estimated)
        var scaled = entries.map {
            Entry(segment: $0.segment, tokens: Int((Double($0.tokens) * scale).rounded()))
        }

        // Rounding each segment independently leaves the sum a few tokens off
        // the exact total. Absorb the drift into the largest segment so the
        // parts still add up to the whole.
        let drift = actual - scaled.reduce(0) { $0 + $1.tokens }
        if drift != 0,
           let largest = scaled.indices.max(by: { scaled[$0].tokens < scaled[$1].tokens }) {
            scaled[largest].tokens += drift
        }

        return ContextUsage(entries: scaled, contextWindow: contextWindow, isEstimated: false)
    }

    /// Renders a token count the way a context readout does: `1.0M`, `499.6k`, `284`.
    public static func format(_ tokens: Int) -> String {
        switch tokens {
        case 1_000_000...:
            String(format: "%.1fM", Double(tokens) / 1_000_000)
        case 1_000...:
            String(format: "%.1fk", Double(tokens) / 1_000)
        default:
            String(tokens)
        }
    }
}
