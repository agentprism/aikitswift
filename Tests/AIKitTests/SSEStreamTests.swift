import Foundation
import Testing

@testable import AIKit

/// The async seam between raw bytes and ``SSEParser``.
///
/// ``SSEParserTests`` feeds the state machine lines that a test invented;
/// these feed it bytes the way a socket does. The framing that matters most —
/// the blank line between frames — only exists at the byte level, so a bridge
/// that eats blank lines passes every parser test and still merges the whole
/// stream into one malformed event.
@Suite("SSE byte stream")
struct SSEStreamTests {

    private func events(_ payload: String) async throws -> [SSEEvent] {
        let bytes = AsyncThrowingStream<UInt8, any Error> { continuation in
            for byte in Array(payload.utf8) {
                continuation.yield(byte)
            }
            continuation.finish()
        }

        var collected: [SSEEvent] = []
        for try await event in sseEvents(from: bytes) {
            collected.append(event)
        }
        return collected
    }

    @Test("each frame arrives as its own event")
    func separatesFrames() async throws {
        let collected = try await events(
            """
            event: message_start
            data: {"type":"message_start"}

            event: content_block_delta
            data: {"type":"content_block_delta"}


            """
        )

        #expect(collected.count == 2)
        #expect(collected.first?.data == #"{"type":"message_start"}"#)
        #expect(collected.last?.data == #"{"type":"content_block_delta"}"#)
    }

    @Test("CRLF framing separates frames too")
    func separatesCRLFFrames() async throws {
        let collected = try await events(
            "data: {\"a\":1}\r\n\r\ndata: {\"a\":2}\r\n\r\n"
        )

        #expect(collected.count == 2)
        #expect(collected.first?.data == #"{"a":1}"#)
        #expect(collected.last?.data == #"{"a":2}"#)
    }

    @Test("no frame carries two JSON objects")
    func neverMergesPayloads() async throws {
        // The failure this guards against is not "an event is missing" but
        // "one event holds every payload", which surfaces downstream as
        // `The given data was not valid JSON` from a wire decoder.
        let collected = try await events(
            "data: {\"i\":0}\n\ndata: {\"i\":1}\n\ndata: {\"i\":2}\n\n"
        )

        #expect(collected.count == 3)
        for event in collected {
            #expect(!event.data.contains("\n"))
            #expect(try JSONSerialization.jsonObject(with: Data(event.data.utf8)) is [String: Any])
        }
    }

    @Test("heartbeats between frames stay invisible")
    func dropsHeartbeats() async throws {
        let collected = try await events(": ping\n\ndata: {\"a\":1}\n\n: ping\n\n")

        #expect(collected.count == 1)
        #expect(collected.first?.data == #"{"a":1}"#)
    }

    @Test("a frame left unterminated still dispatches")
    func flushesTrailingFrame() async throws {
        let collected = try await events("data: {\"a\":1}")

        #expect(collected.count == 1)
        #expect(collected.first?.data == #"{"a":1}"#)
    }
}
