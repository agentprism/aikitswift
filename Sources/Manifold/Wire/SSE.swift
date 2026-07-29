import Foundation

/// A single Server-Sent Event.
public struct SSEEvent: Sendable, Hashable {
    /// The `event:` field. Most LLM APIs also repeat the type inside `data`,
    /// so this is usually redundant — but not always.
    public var event: String?
    /// The `data:` payload. Multiple `data:` lines are joined with newlines,
    /// per the SSE specification.
    public var data: String
    public var id: String?
    public var retry: Int?

    public init(event: String? = nil, data: String, id: String? = nil, retry: Int? = nil) {
        self.event = event
        self.data = data
        self.id = id
        self.retry = retry
    }
}

/// Incremental Server-Sent Events parser.
///
/// Deliberately synchronous: framing bugs are the single most common source of
/// streaming defects, and a pure state machine can be tested exhaustively
/// against recorded bytes with no network, no async, and no provider key.
/// The async plumbing is a thin wrapper over this (see ``sseEvents(from:)``).
public struct SSEParser: Sendable {
    private var dataLines: [String] = []
    private var eventName: String?
    private var eventId: String?
    private var retry: Int?

    public init() {}

    /// Feeds one line (without its trailing newline).
    ///
    /// Returns an event when the line terminates a frame — that is, on a blank
    /// line — and `nil` otherwise.
    public mutating func push(line rawLine: String) -> SSEEvent? {
        // Tolerate CRLF: some proxies rewrite line endings.
        var line = rawLine
        if line.hasSuffix("\r") { line.removeLast() }

        // Blank line dispatches the accumulated frame.
        if line.isEmpty {
            return dispatch()
        }

        // A leading colon marks a comment. Heartbeats arrive this way, and
        // dropping them here is what keeps an idle stream from surfacing
        // spurious events.
        if line.hasPrefix(":") {
            return nil
        }

        let field: String
        let value: String
        if let colon = line.firstIndex(of: ":") {
            field = String(line[line.startIndex..<colon])
            var rest = line[line.index(after: colon)...]
            // Exactly one leading space is stripped, per spec.
            if rest.hasPrefix(" ") { rest = rest.dropFirst() }
            value = String(rest)
        } else {
            // A line with no colon is a field name with an empty value.
            field = line
            value = ""
        }

        switch field {
        case "event": eventName = value
        case "data": dataLines.append(value)
        case "id": eventId = value
        case "retry": retry = Int(value)
        default: break  // Unknown fields are ignored, per spec.
        }

        return nil
    }

    /// Flushes any frame left buffered when the connection ends without a
    /// trailing blank line. Servers do truncate like this.
    public mutating func finish() -> SSEEvent? {
        dispatch()
    }

    private mutating func dispatch() -> SSEEvent? {
        defer {
            dataLines.removeAll()
            eventName = nil
            retry = nil
        }

        // A frame with no data lines is not an event.
        guard !dataLines.isEmpty else { return nil }

        return SSEEvent(
            event: eventName,
            data: dataLines.joined(separator: "\n"),
            id: eventId,
            retry: retry
        )
    }
}

// MARK: - Async bridge

/// Wraps a byte stream — typically `URLSession.bytes(for:).0` — as a stream of
/// parsed Server-Sent Events.
public func sseEvents<Bytes: AsyncSequence & Sendable>(
    from bytes: Bytes
) -> AsyncThrowingStream<SSEEvent, any Error> where Bytes.Element == UInt8 {
    AsyncThrowingStream { continuation in
        let task = Task {
            var parser = SSEParser()
            do {
                for try await line in bytes.lines {
                    if let event = parser.push(line: line) {
                        continuation.yield(event)
                    }
                }
                if let event = parser.finish() {
                    continuation.yield(event)
                }
                continuation.finish()
            } catch {
                continuation.finish(throwing: error)
            }
        }
        continuation.onTermination = { _ in task.cancel() }
    }
}
