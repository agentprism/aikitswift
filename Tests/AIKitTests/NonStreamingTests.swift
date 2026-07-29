import Foundation
import Testing

@testable import AIKit

extension Fixture {
    /// Complete (non-streamed) response bodies recorded alongside the streams.
    ///
    /// Filtered by shape rather than by filename: a set's `.json` files also
    /// include request fixtures and other artefacts, and a body that is not a
    /// response would fail for reasons that say nothing about the decoder.
    static func responseBodies(_ set: String, matching isResponse: (JSONValue) -> Bool) throws -> [(String, JSONValue)] {
        try FileManager.default
            .contentsOfDirectory(at: try directory(set), includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" && $0.lastPathComponent != "PROVENANCE.md" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
            .compactMap { url in
                guard let data = try? Data(contentsOf: url),
                      let value = try? JSONValue.decode(from: data),
                      isResponse(value)
                else { return nil }
                return (url.deletingPathExtension().lastPathComponent, value)
            }
    }
}

/// A complete response body must decode to the same normalized shape a stream
/// would have produced.
///
/// This is enforced structurally: the decoder replays a body as the events a
/// stream would deliver and runs them through the same mapper, so the two paths
/// cannot drift. These tests check that the replay is faithful.
@Suite("Non-streaming responses")
struct NonStreamingTests {

    static var anthropicBodies: [String] {
        get throws {
            try Fixture.responseBodies("anthropic") { $0["type"]?.stringValue == "message" }
                .map(\.0)
        }
    }

    static var responsesBodies: [String] {
        get throws {
            try Fixture.responseBodies("openai-responses") { $0["output"]?.arrayValue != nil }
                .map(\.0)
        }
    }

    static var googleBodies: [String] {
        get throws {
            try Fixture.responseBodies("google") { $0["candidates"]?.arrayValue != nil }
                .map(\.0)
        }
    }

    private static func body(_ set: String, _ name: String) throws -> JSONValue {
        let url = try Fixture.directory(set).appending(path: "\(name).json")
        return try JSONValue.decode(from: try Data(contentsOf: url))
    }

    @Test("response bodies are present")
    func bodiesArePresent() throws {
        #expect(try Self.anthropicBodies.count >= 20)
        #expect(try Self.responsesBodies.count >= 5)
    }

    // MARK: - Conformance

    @Test("Anthropic bodies decode to a well-formed stream", arguments: try anthropicBodies)
    func anthropicConforms(name: String) throws {
        let parts = NonStreamingResponse.decode(try Self.body("anthropic", name), wire: .anthropicMessages)
        WireConformance.check([parts], label: name)
    }

    @Test("Responses bodies decode to a well-formed stream", arguments: try responsesBodies)
    func responsesConform(name: String) throws {
        let parts = NonStreamingResponse.decode(try Self.body("openai-responses", name), wire: .openAIResponses)
        WireConformance.check([parts], label: name)
    }

    @Test("Google bodies decode to a well-formed stream", arguments: try googleBodies)
    func googleConforms(name: String) throws {
        let parts = NonStreamingResponse.decode(try Self.body("google", name), wire: .googleGenerativeAI)
        WireConformance.check([parts], label: name)
    }

    // MARK: - Fidelity

    @Test("decoded text matches the body verbatim", arguments: try anthropicBodies)
    func preservesText(name: String) throws {
        let body = try Self.body("anthropic", name)
        let parts = NonStreamingResponse.decode(body, wire: .anthropicMessages)

        let expected = (body["content"]?.arrayValue ?? [])
            .filter { $0["type"]?.stringValue == "text" }
            .compactMap { $0["text"]?.stringValue }
            .joined()

        let decoded = parts.compactMap {
            if case .textDelta(_, let delta, _) = $0 { return delta } else { return nil }
        }.joined()

        #expect(decoded == expected, "\(name): text was altered in transit")
    }

    @Test("decoded tool calls match the body", arguments: try anthropicBodies)
    func preservesToolCalls(name: String) throws {
        let body = try Self.body("anthropic", name)
        let parts = NonStreamingResponse.decode(body, wire: .anthropicMessages)

        let expected = (body["content"]?.arrayValue ?? [])
            .filter { $0["type"]?.stringValue == "tool_use" }
            .compactMap { $0["id"]?.stringValue }

        let decoded = parts.compactMap {
            if case .toolCall(let call) = $0, !call.providerExecuted { return call.toolCallId }
            return nil
        }

        #expect(Set(decoded) == Set(expected), "\(name): tool calls lost or invented")
    }

    @Test("usage survives the round trip", arguments: try anthropicBodies)
    func preservesUsage(name: String) throws {
        let body = try Self.body("anthropic", name)
        let parts = NonStreamingResponse.decode(body, wire: .anthropicMessages)

        guard case .finish(let usage, _, _)? = parts.last else {
            Issue.record("\(name): no terminal finish")
            return
        }

        // Input counts arrive at `message_start` and output counts at
        // `message_delta`; a synthesis that put them in one place would let the
        // mapper's two-phase accumulation silently drop half.
        let providerUsage = body["usage"] ?? .null
        let iterations = providerUsage["iterations"]?.arrayValue ?? []
        let servedByFallback = iterations.contains { $0["type"]?.stringValue == "fallback_message" }

        guard iterations.isEmpty || servedByFallback else {
            // With compaction or the advisor tool in play the top-level totals
            // deliberately exclude some iterations, so they are *not* what the
            // caller was billed. Asserting equality here would enshrine the
            // under-report this library exists to avoid.
            let executorInput = iterations
                .filter { ["compaction", "message"].contains($0["type"]?.stringValue ?? "") }
                .reduce(0) { $0 + ($1["input_tokens"]?.intValue ?? 0) }

            #expect(
                usage.inputTokens.noCache == executorInput,
                "\(name): executor iterations were not summed"
            )
            #expect(
                (usage.inputTokens.noCache ?? 0) >= (providerUsage["input_tokens"]?.intValue ?? 0),
                "\(name): summed usage is below the top-level figure"
            )
            return
        }

        #expect(
            usage.inputTokens.noCache == providerUsage["input_tokens"]?.intValue,
            "\(name): input tokens lost"
        )
        #expect(
            usage.outputTokens.total == providerUsage["output_tokens"]?.intValue,
            "\(name): output tokens lost"
        )
    }

    @Test("stop reasons survive the round trip", arguments: try anthropicBodies)
    func preservesStopReason(name: String) throws {
        let body = try Self.body("anthropic", name)
        let parts = NonStreamingResponse.decode(body, wire: .anthropicMessages)

        guard case .finish(_, let reason, _)? = parts.last else { return }
        #expect(reason.raw == body["stop_reason"]?.stringValue, "\(name): stop reason lost")
    }

    @Test("thinking signatures survive the round trip")
    func preservesThinkingSignature() {
        // The signature must be replayed to the API unchanged on the next turn
        // or the request is rejected, so losing it breaks multi-turn reasoning.
        let body: JSONValue = [
            "type": "message", "id": "msg_1", "model": "claude-opus-4-8", "role": "assistant",
            "content": [[
                "type": "thinking",
                "thinking": "considering",
                "signature": "sig_abc123",
            ]],
            "stop_reason": "end_turn",
            "usage": ["input_tokens": 10, "output_tokens": 5],
        ]

        let parts = NonStreamingResponse.decode(body, wire: .anthropicMessages)
        let signature = parts.compactMap {
            if case .reasoningDelta(_, _, let metadata) = $0 {
                return metadata?["anthropic"]?["signature"]?.stringValue
            }
            return nil
        }.first

        #expect(signature == "sig_abc123")
    }

    @Test("a Gemini body needs no rewriting")
    func geminiBodyIsAlreadyChunkShaped() {
        // Gemini's non-streaming body has the same shape as a stream chunk,
        // which is why the synthesizer passes it straight through.
        let body: JSONValue = [
            "candidates": [[
                "index": 0,
                "content": ["parts": [["text": "hello"]], "role": "model"],
                "finishReason": "STOP",
            ]],
            "usageMetadata": ["promptTokenCount": 5, "candidatesTokenCount": 2],
            "modelVersion": "gemini-3-pro",
        ]

        #expect(NonStreamingResponse.synthesize(body, wire: .googleGenerativeAI).count == 1)

        let parts = NonStreamingResponse.decode(body, wire: .googleGenerativeAI)
        let text = parts.compactMap {
            if case .textDelta(_, let delta, _) = $0 { return delta } else { return nil }
        }.joined()
        #expect(text == "hello")
    }

    @Test("a Chat Completions body decodes")
    func decodesCompletionsBody() {
        let body: JSONValue = [
            "id": "chatcmpl-1", "object": "chat.completion", "model": "gpt-5",
            "choices": [[
                "index": 0,
                "message": ["role": "assistant", "content": "hello"],
                "finish_reason": "stop",
            ]],
            "usage": ["prompt_tokens": 10, "completion_tokens": 2],
        ]

        let parts = NonStreamingResponse.decode(body, wire: .openAICompletions)
        WireConformance.check([parts], label: "completions-body")

        let text = parts.compactMap {
            if case .textDelta(_, let delta, _) = $0 { return delta } else { return nil }
        }.joined()
        #expect(text == "hello")

        guard case .finish(let usage, let reason, _)? = parts.last else {
            Issue.record("no terminal finish")
            return
        }
        #expect(usage.inputTokens.total == 10)
        #expect(reason.unified == .stop)
    }

    @Test("provider-executed results survive a non-streamed body")
    func preservesServerToolResults() {
        let body: JSONValue = [
            "type": "message", "id": "msg_1", "model": "claude-opus-4-8", "role": "assistant",
            "content": [
                ["type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": ["query": "swift"]],
                ["type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
                    ["type": "web_search_result", "url": "https://swift.org", "title": "Swift"]
                ]],
            ],
            "stop_reason": "end_turn",
            "usage": ["input_tokens": 10, "output_tokens": 5],
        ]

        let parts = NonStreamingResponse.decode(body, wire: .anthropicMessages)
        WireConformance.check([parts], label: "server-tool-body")

        #expect(parts.contains {
            if case .toolCall(let call) = $0 { return call.providerExecuted }
            return false
        })
        #expect(parts.contains { if case .toolResult = $0 { return true } else { return false } })
        #expect(parts.contains { if case .source = $0 { return true } else { return false } })
    }
}
