import Foundation
import Testing

@testable import AIKit

@Suite("Transcript codable support")
struct CodableSupportTests {

    @Test("messages round-trip across every content part kind")
    func messageRoundTrip() throws {
        let file = FilePart(
            mediaType: "text/plain",
            data: .url("https://example.com/file.txt"),
            filename: "file.txt"
        )
        let reasoningMetadata: ProviderMetadata = [
            "anthropic": [
                "signature": .string("sig_123")
            ]
        ]
        let toolCall = ToolCall(
            toolCallId: "call_1",
            toolName: "search_health",
            namespace: "health",
            input: #"{"days":7}"#,
            providerExecuted: true,
            dynamic: true,
            providerMetadata: [
                "openai": ["source": .string("mcp")]
            ]
        )
        let toolResult = ToolResult(
            toolCallId: "call_1",
            toolName: "search_health",
            result: .object(["summary": "ok"]),
            content: [
                .text("ok"),
                .file(FilePart(mediaType: "image/png", data: .base64("cG5n"))),
            ],
            isError: false,
            preliminary: true,
            dynamic: true,
            providerMetadata: [
                "openai": ["result_type": .string("structured")]
            ]
        )
        let message = Message(
            role: .assistant,
            content: [
                .text("done"),
                .file(file),
                .reasoning("thinking", providerMetadata: reasoningMetadata),
                .toolCall(toolCall),
                .toolResult(toolResult),
            ],
            providerOptions: [
                "anthropic": ["cache_control": .string("ephemeral")]
            ]
        )

        let decoded = try roundTrip(message)
        #expect(decoded == message)
    }

    @Test("tool call decoding tolerates newly-added required flags being absent in old data")
    func toolCallLegacyDecode() throws {
        let data = try JSONSerialization.data(withJSONObject: [
            "toolCallId": "call_legacy",
            "toolName": "lookup",
            "input": #"{"query":"sleep"}"#
        ])

        let call = try JSONDecoder().decode(ToolCall.self, from: data)
        #expect(call.toolCallId == "call_legacy")
        #expect(call.providerExecuted == false)
        #expect(call.dynamic == false)
    }

    @Test("tool result decoding tolerates old payloads without status flags")
    func toolResultLegacyDecode() throws {
        let data = try JSONSerialization.data(withJSONObject: [
            "toolCallId": "call_legacy",
            "toolName": "lookup",
            "result": ["summary": "ok"]
        ])

        let result = try JSONDecoder().decode(ToolResult.self, from: data)
        #expect(result.toolName == "lookup")
        #expect(result.isError == false)
        #expect(result.preliminary == false)
        #expect(result.dynamic == false)
    }

    @Test("finish reason and usage round-trip")
    func finishReasonAndUsageRoundTrip() throws {
        let finishReason = FinishReason(unified: .length, raw: "max_tokens")
        let usage = Usage(
            inputTokens: .init(total: 12_345, noCache: 12_000, cacheRead: 300, cacheWrite: 45),
            outputTokens: .init(total: 678, text: 500, reasoning: 178),
            raw: .object(["tier": "pro"])
        )

        #expect(try roundTrip(finishReason) == finishReason)
        #expect(try roundTrip(usage) == usage)
    }

    private func roundTrip<T: Codable & Equatable>(_ value: T) throws -> T {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        return try decoder.decode(T.self, from: encoder.encode(value))
    }
}
