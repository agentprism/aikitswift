import Foundation
import Testing

@testable import AIKit

/// The aggregate view over a part sequence, and the multi-turn replay message
/// built from it.
@Suite("AIResponse")
struct AIResponseTests {

    /// A response with interleaved blocks: reasoning, then text, then a tool
    /// call, the way a thinking model actually answers.
    private var interleaved: AIResponse {
        AIResponse(parts: [
            .streamStart(warnings: [Warning(message: "temperature dropped", setting: "temperature")]),
            .responseMetadata(ResponseMetadata(id: "msg_1", modelId: "test-model")),
            .reasoningStart(id: "0"),
            .reasoningDelta(id: "0", delta: "Let me "),
            .reasoningDelta(id: "0", delta: "think."),
            .reasoningDelta(id: "0", delta: "", providerMetadata: ["anthropic": ["signature": .string("sig-bytes")]]),
            .reasoningEnd(id: "0"),
            .textStart(id: "1"),
            .textDelta(id: "1", delta: "Checking "),
            .textDelta(id: "1", delta: "the weather."),
            .textEnd(id: "1"),
            .toolInputStart(id: "call_1", toolName: "get_weather"),
            .toolInputDelta(id: "call_1", delta: #"{"city":"#),
            .toolInputDelta(id: "call_1", delta: #""Paris"}"#),
            .toolInputEnd(id: "call_1"),
            .toolCall(ToolCall(toolCallId: "call_1", toolName: "get_weather", input: #"{"city":"Paris"}"#)),
            .finish(
                usage: Usage(inputTokens: .init(total: 10), outputTokens: .init(total: 20)),
                finishReason: FinishReason(unified: .toolCalls, raw: "tool_use")
            ),
        ])
    }

    @Test("aggregates text, reasoning and tool calls from the parts")
    func aggregates() {
        let response = interleaved

        #expect(response.text == "Checking the weather.")
        #expect(response.reasoning == "Let me think.")
        #expect(response.toolCalls.map(\.toolCallId) == ["call_1"])
        #expect(response.pendingToolCalls.map(\.toolName) == ["get_weather"])
        #expect(response.warnings.map(\.setting) == ["temperature"])
        #expect(response.usage.inputTokens.total == 10)
        #expect(response.finishReason?.unified == .toolCalls)
        #expect(response.metadata?.id == "msg_1")
    }

    @Test("assistantMessage keeps block order and the reasoning signature")
    func assistantMessagePreservesOrderAndSignature() {
        let message = interleaved.assistantMessage

        #expect(message.role == .assistant)
        guard message.content.count == 3 else {
            Issue.record("expected reasoning, text, toolCall — got \(message.content)")
            return
        }

        // Order matters: Anthropic rejects a replayed turn whose thinking
        // block does not come first.
        guard case .reasoning(let thinking, let metadata) = message.content[0] else {
            Issue.record("first part should be reasoning")
            return
        }
        #expect(thinking == "Let me think.")
        #expect(metadata?["anthropic"]?["signature"] == .string("sig-bytes"))

        #expect(message.content[1] == .text("Checking the weather."))

        guard case .toolCall(let call) = message.content[2] else {
            Issue.record("third part should be the tool call")
            return
        }
        #expect(call.input == #"{"city":"Paris"}"#)
    }

    @Test("provider-executed tool calls are not replayed")
    func omitsProviderExecutedCalls() {
        let response = AIResponse(parts: [
            .toolCall(ToolCall(
                toolCallId: "srv_1", toolName: "web_search",
                input: "{}", providerExecuted: true
            )),
            .toolCall(ToolCall(toolCallId: "call_1", toolName: "local", input: "{}")),
        ])

        #expect(response.toolCalls.count == 2)
        #expect(response.pendingToolCalls.map(\.toolCallId) == ["call_1"])
        #expect(response.assistantMessage.content == [
            .toolCall(ToolCall(toolCallId: "call_1", toolName: "local", input: "{}"))
        ])
    }

    @Test("a finish-less response reports nil, not a fabricated reason")
    func incompleteResponse() {
        let response = AIResponse(parts: [
            .textStart(id: "0"),
            .textDelta(id: "0", delta: "cut off mid-"),
        ])

        #expect(response.finishReason == nil)
        #expect(response.usage == .empty)
        #expect(response.text == "cut off mid-")
    }

    @Test("collect() drains a stream into the same aggregate")
    func collectsAStream() async throws {
        let source = interleaved.parts
        let stream = AsyncThrowingStream<StreamPart, any Error> { continuation in
            for part in source { continuation.yield(part) }
            continuation.finish()
        }

        let response = try await stream.collect()
        #expect(response == interleaved)
    }

    @Test("a recorded thinking stream survives the round trip back into a request")
    func fixtureRoundTrip() throws {
        // The strongest claim this type makes: the assistant message it builds
        // from a real recorded stream, when encoded into the next request,
        // still carries the thinking signature byte-for-byte. Hand-built
        // replays losing the signature is the classic multi-turn failure.
        let parts = try #require(
            try Fixture.replay(
                AnthropicMessagesWire.self, "anthropic", "anthropic-clear-thinking.1",
                splitOn: Fixture.anthropicBoundary
            ).first
        )
        let response = AIResponse(parts: parts)
        let signatures = response.assistantMessage.content.compactMap { part -> JSONValue? in
            if case .reasoning(_, let metadata) = part {
                return metadata?["anthropic"]?["signature"]
            }
            return nil
        }
        let signature = try #require(signatures.first)

        var prompt: Prompt = [.user("original question")]
        prompt.append(response.assistantMessage)
        let body = AnthropicMessagesRequest.encode(
            CallOptions(model: "claude-opus-4-8", prompt: prompt)
        ).body

        let encodedThinking = try #require(
            body["messages"]?.arrayValue?.last?["content"]?.arrayValue?
                .first { $0["type"]?.stringValue == "thinking" }
        )
        #expect(encodedThinking["signature"] == signature)
    }
}

/// `generate()` sends a genuinely non-streaming request; these pin the ways
/// its wire shape must differ from the streaming one.
@Suite("Non-streaming requests")
struct NonStreamingRequestTests {

    private let options = CallOptions(model: "m", prompt: [.user("hi")])

    @Test("the stream flag is absent, not false")
    func omitsStreamFlag() {
        #expect(AnthropicMessagesRequest.encode(options, streaming: false).body["stream"] == nil)
        #expect(OpenAIResponsesRequest.encode(options, streaming: false).body["stream"] == nil)
        #expect(OpenAICompletionsRequest.encode(options, streaming: false).body["stream"] == nil)

        // And the default remains streaming; existing call sites are unchanged.
        #expect(AnthropicMessagesRequest.encode(options).body["stream"] == .bool(true))
    }

    @Test("stream_options never rides on a non-streaming request")
    func omitsStreamOptions() {
        // `stream_options` is only legal alongside `stream: true`; sending it
        // on a non-streaming request is a 400 on the reference API.
        var dialect = CompletionsDialect.default
        dialect.supportsUsageInStreaming = true

        let streamed = OpenAICompletionsRequest.encode(options, dialect: dialect).body
        #expect(streamed["stream_options"] != nil)

        let complete = OpenAICompletionsRequest.encode(options, dialect: dialect, streaming: false).body
        #expect(complete["stream_options"] == nil)
    }

    @Test("Gemini switches endpoint method instead of a body flag")
    func geminiEndpoint() throws {
        let subject = try AIClient(providerId: "google", configuration: .init(apiKey: "k"))
        let url = subject.endpoint(
            wire: .googleGenerativeAI,
            base: URL(string: "https://generativelanguage.googleapis.com")!,
            model: "gemini-3-pro",
            streaming: false
        )

        #expect(url.absoluteString.contains("gemini-3-pro:generateContent"))
        #expect(!url.absoluteString.contains("streamGenerateContent"))
        #expect(!url.absoluteString.contains("alt=sse"))
    }

    @Test("the accept header asks for JSON, not an event stream")
    func acceptHeader() throws {
        let subject = try AIClient(providerId: "openai", configuration: .init(apiKey: "k"))
        let body = subject.encode(options, wire: .openAICompletions, model: nil, streaming: false).body

        let request = try subject.makeRequest(
            wire: .openAICompletions, options: options, body: body, streaming: false
        )
        #expect(request.value(forHTTPHeaderField: "accept") == "application/json")
    }
}
