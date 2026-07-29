import Foundation

/// Why the model stopped generating.
///
/// Carries both a normalized reason (so callers can branch portably) and the
/// provider's original string (so nothing is lost). Providers keep inventing
/// new stop reasons; `raw` is how you see one that `unified` flattened away.
public struct FinishReason: Sendable, Hashable {
    public enum Unified: String, Sendable, Hashable, Codable {
        /// The model finished on its own.
        case stop
        /// The output token cap was reached.
        case length
        /// A safety classifier declined the request.
        case contentFilter = "content-filter"
        /// The model wants one or more tools executed.
        case toolCalls = "tool-calls"
        /// Generation failed.
        case error
        case other
    }

    public var unified: Unified
    /// The provider's own stop reason, verbatim.
    public var raw: String?

    public init(unified: Unified, raw: String? = nil) {
        self.unified = unified
        self.raw = raw
    }

    public static let stop = FinishReason(unified: .stop)
    public static let other = FinishReason(unified: .other)
}
