import Foundation

/// A streaming wire protocol implementation.
///
/// One conformance per *protocol*, not per provider. Providers that speak the
/// same protocol share an implementation and differ only in configuration —
/// which is why the catalog holds fifty providers but only a handful of these.
///
/// Conformances are state machines over a single response. Create one per
/// stream; never share one across requests.
public protocol WireMapper: Sendable {
    init()

    /// Maps one decoded provider chunk to zero or more normalized parts.
    mutating func map(chunk: JSONValue) -> [StreamPart]

    /// Maps one raw SSE `data:` payload.
    mutating func map(rawJSON: String) -> [StreamPart]

    /// Closes the stream, emitting anything the protocol defers to the end.
    ///
    /// Must be idempotent, and must be called even when the connection drops:
    /// several protocols carry final usage nowhere else, and a mapper that is
    /// never closed silently reports no cost at all.
    mutating func finish() -> [StreamPart]
}
