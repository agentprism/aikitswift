import Foundation
import Testing

@testable import AIKit

/// One pass over the whole streaming path: URLSession bytes → SSE framing →
/// mapper → ``StreamPart``.
///
/// Every other suite tests a layer in isolation, which is how a stream that
/// decoded perfectly in the mapper tests still arrived as a single malformed
/// event over a socket. This suite is the seam check: it stubs the transport,
/// not the framing.
/// Serialized: the stub's canned response is process-wide state, and tests in a
/// suite otherwise run concurrently.
@Suite("Streaming over HTTP", .serialized)
struct StreamingIntegrationTests {

    private func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [StubProtocol.self]
        return URLSession(configuration: configuration)
    }

    private func client(_ providerId: String) throws -> AIClient {
        try AIClient(
            providerId: providerId,
            configuration: .init(apiKey: "sk-test"),
            session: session()
        )
    }

    private var options: CallOptions {
        CallOptions(model: "claude-sonnet-5", prompt: [.user("hi")])
    }

    /// A recorded Anthropic reply, framed the way the API frames it.
    private let anthropicStream = """
        event: message_start
        data: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"claude-sonnet-5","content":[],"usage":{"input_tokens":10,"output_tokens":1}}}

        event: content_block_start
        data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

        event: content_block_delta
        data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"你最近"}}

        event: content_block_delta
        data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"睡得不错。"}}

        event: content_block_stop
        data: {"type":"content_block_stop","index":0}

        event: message_delta
        data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}

        event: message_stop
        data: {"type":"message_stop"}


        """

    @Test("text arrives as separate deltas, not one merged chunk")
    func streamsTextDeltas() async throws {
        StubProtocol.serve(status: 200, body: anthropicStream)

        var deltas: [String] = []
        var finished = false
        for try await part in try client("anthropic").stream(options) {
            switch part {
            case .textDelta(_, let delta, _): deltas.append(delta)
            case .finish: finished = true
            default: break
            }
        }

        #expect(deltas == ["你最近", "睡得不错。"])
        #expect(finished)
    }

    @Test("CRLF framing survives the transport")
    func streamsThroughCRLFFraming() async throws {
        // Proxies rewrite line endings; the payload is unchanged.
        StubProtocol.serve(
            status: 200,
            body: anthropicStream.replacingOccurrences(of: "\n", with: "\r\n")
        )

        var text = ""
        for try await part in try client("anthropic").stream(options) {
            if case .textDelta(_, let delta, _) = part { text += delta }
        }

        #expect(text == "你最近睡得不错。")
    }

    @Test("an error body reaches the caller verbatim")
    func surfacesErrorBody() async throws {
        // The provider's own words are the only diagnostic a caller gets, so
        // the body travels whole — line breaks included.
        let body = "{\n  \"error\": {\"message\": \"invalid x-api-key\"}\n}"
        StubProtocol.serve(status: 401, body: body)

        await #expect(throws: AIClientError.self) {
            for try await _ in try self.client("anthropic").stream(self.options) {}
        }

        do {
            for try await _ in try client("anthropic").stream(options) {}
            Issue.record("expected the request to fail")
        } catch let error as AIClientError {
            guard case .http(let status, let received) = error.kind else {
                Issue.record("expected an HTTP error, got \(error.kind)")
                return
            }
            #expect(status == 401)
            #expect(received == body)
        }
    }
}

/// Serves a canned response to any request. Bodies are handed over whole; the
/// framing under test is inside them, not in how many pieces they arrive in.
private final class StubProtocol: URLProtocol {
    nonisolated(unsafe) private static var status = 200
    nonisolated(unsafe) private static var body = Data()
    private static let lock = NSLock()

    static func serve(status: Int, body: String) {
        lock.lock()
        defer { lock.unlock() }
        Self.status = status
        Self.body = Data(body.utf8)
    }

    private static var response: (status: Int, body: Data) {
        lock.lock()
        defer { lock.unlock() }
        return (status, body)
    }

    override class func canInit(with request: URLRequest) -> Bool { true }

    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let canned = Self.response
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: canned.status,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": "text/event-stream"]
        )!

        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: canned.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
