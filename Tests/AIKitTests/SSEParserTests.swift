import Testing

@testable import AIKit

@Suite("SSE framing")
struct SSEParserTests {

    /// Feeds a whole payload line by line, the way a byte stream would.
    private func parse(_ payload: String) -> [SSEEvent] {
        var parser = SSEParser()
        var events: [SSEEvent] = []
        for line in payload.components(separatedBy: "\n") {
            if let event = parser.push(line: line) { events.append(event) }
        }
        if let event = parser.finish() { events.append(event) }
        return events
    }

    @Test("blank line dispatches a frame")
    func dispatchesOnBlankLine() {
        let events = parse("event: message_start\ndata: {\"a\":1}\n\n")

        #expect(events.count == 1)
        #expect(events[0].event == "message_start")
        #expect(events[0].data == "{\"a\":1}")
    }

    @Test("multiple data lines join with newlines")
    func joinsMultipleDataLines() {
        let events = parse("data: line one\ndata: line two\n\n")

        #expect(events.count == 1)
        #expect(events[0].data == "line one\nline two")
    }

    @Test("exactly one leading space is stripped from a value")
    func stripsOneLeadingSpace() {
        // The second space is part of the payload, not the framing.
        let events = parse("data:  padded\n\n")

        #expect(events[0].data == " padded")
    }

    @Test("comment lines are ignored")
    func ignoresComments() {
        // Heartbeats arrive as bare comments. They must not surface as events,
        // or an idle stream looks like it is producing output.
        let events = parse(": keep-alive\ndata: real\n\n")

        #expect(events.count == 1)
        #expect(events[0].data == "real")
    }

    @Test("a comment-only frame produces nothing")
    func heartbeatOnlyProducesNothing() {
        #expect(parse(": ping\n\n: ping\n\n").isEmpty)
    }

    @Test("CRLF line endings are tolerated")
    func toleratesCRLF() {
        // Some proxies rewrite line endings; the carriage return must not end
        // up inside the payload.
        let events = parse("event: ping\r\ndata: {\"x\":1}\r\n\r\n")

        #expect(events.count == 1)
        #expect(events[0].event == "ping")
        #expect(events[0].data == "{\"x\":1}")
    }

    @Test("a frame with no data lines is not an event")
    func requiresData() {
        #expect(parse("event: lonely\n\n").isEmpty)
    }

    @Test("a truncated final frame is still delivered")
    func flushesOnFinish() {
        // Servers do close without a trailing blank line. Dropping the last
        // frame would silently truncate the response.
        let events = parse("data: {\"last\":true}")

        #expect(events.count == 1)
        #expect(events[0].data == "{\"last\":true}")
    }

    @Test("unknown fields are ignored")
    func ignoresUnknownFields() {
        let events = parse("weird: value\ndata: kept\n\n")

        #expect(events.count == 1)
        #expect(events[0].data == "kept")
    }

    @Test("a field with no colon has an empty value")
    func handlesValuelessField() {
        let events = parse("data\ndata: after\n\n")

        #expect(events[0].data == "\nafter")
    }

    @Test("consecutive frames are independent")
    func resetsBetweenFrames() {
        let events = parse("event: first\ndata: 1\n\ndata: 2\n\n")

        #expect(events.count == 2)
        #expect(events[0].event == "first")
        // The event name must not leak into the next frame.
        #expect(events[1].event == nil)
        #expect(events[1].data == "2")
    }
}
