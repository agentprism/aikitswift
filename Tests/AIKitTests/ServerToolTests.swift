import Foundation
import Testing

@testable import AIKit

/// Provider-executed tools — web search, code execution, MCP — report through
/// blocks the client must never run itself.
///
/// Getting this wrong is worse than dropping it: a server tool call mistaken
/// for a client one gets executed a second time, which for a shell or
/// apply-patch tool means real side effects.
@Suite("Server-executed tools")
struct ServerToolTests {

    @Test("server tool calls are flagged as provider-executed")
    func flagsProviderExecuted() throws {
        let calls = try Fixture.replay(
            AnthropicMessagesWire.self, "anthropic", "anthropic-web-search-tool.1"
        )
        .flatMap { $0 }
        .compactMap { if case .toolCall(let call) = $0 { return call } else { return nil } }

        let searchCalls = calls.filter { $0.toolName == "web_search" }
        #expect(!searchCalls.isEmpty, "expected a web_search call")
        #expect(
            searchCalls.allSatisfy { $0.providerExecuted },
            "a provider-executed call marked as client-executed would be run twice"
        )
    }

    @Test("client tool calls are not flagged as provider-executed")
    func doesNotOverFlag() throws {
        // The inverse mistake: refusing to run a tool the client owns.
        let calls = try Fixture.replay(
            AnthropicMessagesWire.self, "anthropic", "anthropic-json-tool.1",
            splitOn: Fixture.anthropicBoundary
        )
        .flatMap { $0 }
        .compactMap { if case .toolCall(let call) = $0 { return call } else { return nil } }

        #expect(calls.allSatisfy { !$0.providerExecuted })
    }

    @Test("server tool results are structured, not raw")
    func emitsStructuredResults() throws {
        let parts = try Fixture.replay(
            AnthropicMessagesWire.self, "anthropic", "anthropic-web-search-tool.1"
        ).flatMap { $0 }

        let results = parts.compactMap {
            if case .toolResult(let result) = $0 { return result } else { return nil }
        }

        #expect(!results.isEmpty, "web search produced no tool result")
        // The result must reference the call it answers, or nothing downstream
        // can pair them.
        #expect(results.allSatisfy { !$0.toolCallId.isEmpty })
        #expect(results.allSatisfy { !$0.toolName.isEmpty })
    }

    @Test("results carry the id of the call they answer")
    func pairsResultsWithCalls() throws {
        let parts = try Fixture.replay(
            AnthropicMessagesWire.self, "anthropic", "anthropic-web-search-tool.1"
        ).flatMap { $0 }

        var callIds: Set<String> = []
        for part in parts {
            switch part {
            case .toolCall(let call): callIds.insert(call.toolCallId)
            case .toolResult(let result):
                #expect(
                    callIds.contains(result.toolCallId),
                    "result \(result.toolCallId) answers no call seen so far"
                )
            default: break
            }
        }
    }

    @Test("web search results are also emitted as citable sources")
    func emitsSources() throws {
        // A caller should be able to render attributions without knowing which
        // provider ran the search or how it shapes its payload.
        let sources = try Fixture.replay(
            AnthropicMessagesWire.self, "anthropic", "anthropic-web-search-tool.1"
        )
        .flatMap { $0 }
        .compactMap { if case .source(let source) = $0 { return source } else { return nil } }

        #expect(!sources.isEmpty, "web search produced no sources")
        #expect(sources.allSatisfy {
            if case .url(let url) = $0.kind { return url.hasPrefix("http") }
            return false
        })
    }

    @Test("MCP tools are marked dynamic")
    func marksMCPToolsDynamic() throws {
        // MCP tools are defined at runtime, so a caller cannot validate them
        // against a declared schema. `dynamic` is how that is signalled.
        let calls = try Fixture.replay(
            AnthropicMessagesWire.self, "anthropic", "anthropic-mcp.1"
        )
        .flatMap { $0 }
        .compactMap { if case .toolCall(let call) = $0 { return call } else { return nil } }

        #expect(!calls.isEmpty)
        #expect(calls.contains { $0.dynamic && $0.providerExecuted })
    }

    @Test("code execution results are surfaced", arguments: [
        "anthropic-code-execution-20250825.1",
        "anthropic-code-execution-20250825.2",
    ])
    func surfacesCodeExecution(name: String) throws {
        let parts = try Fixture.replay(AnthropicMessagesWire.self, "anthropic", name)
            .flatMap { $0 }

        let results = parts.compactMap {
            if case .toolResult(let result) = $0 { return result } else { return nil }
        }
        #expect(!results.isEmpty, "\(name): code execution produced no structured result")

        // The payload shape is tool-specific and is passed through intact
        // rather than flattened into a lowest common denominator.
        #expect(results.allSatisfy { !$0.result.isNull })
    }

    @Test("server tool payloads survive intact")
    func preservesPayloads() throws {
        let results = try Fixture.replay(
            AnthropicMessagesWire.self, "anthropic", "anthropic-web-search-tool.1"
        )
        .flatMap { $0 }
        .compactMap { if case .toolResult(let result) = $0 { return result } else { return nil } }

        // Search payloads carry titles, URLs and encrypted blobs that only the
        // provider can interpret. Losing them would make citations impossible.
        let first = try #require(results.first)
        #expect(first.result.arrayValue?.isEmpty == false)
        #expect(first.result[0]?["url"]?.stringValue != nil)
    }

    @Test("every recorded stream still conforms", arguments: try Fixture.anthropicStreamNamesForServerTools)
    func stillConforms(name: String) throws {
        // Promoting blocks out of `.raw` must not unbalance any triad.
        WireConformance.check(
            try Fixture.replay(
                AnthropicMessagesWire.self, "anthropic", name,
                splitOn: Fixture.anthropicBoundary
            ),
            label: name
        )
    }
}

extension Fixture {
    /// Recordings that exercise at least one server-executed tool.
    static var anthropicStreamNamesForServerTools: [String] {
        get throws {
            try streamNames("anthropic").filter { name in
                let chunks = (try? chunks("anthropic", name)) ?? []
                return chunks.contains { chunk in
                    guard let type = chunk["content_block"]?["type"]?.stringValue else { return false }
                    return type.hasSuffix("_tool_result") || type == "server_tool_use" || type == "mcp_tool_use"
                }
            }
        }
    }
}

/// Provider-executed tools on the other two protocols.
///
/// The Gemini cases are synthetic: the vendored fixture corpus contains no
/// recording that exercises code execution or grounding, so these encode the
/// documented shapes rather than captured traffic. That is a weaker guarantee
/// than the Anthropic and Responses suites, and is called out rather than
/// papered over.
@Suite("Server tools — other protocols")
struct OtherProtocolServerToolTests {

    @Test("Responses server tools are provider-executed", arguments: [
        "openai-web-search-tool.1",
        "openai-code-interpreter-tool.1",
        "openai-apply-patch-tool.1",
    ])
    func responsesFlagsProviderExecuted(name: String) throws {
        let parts = try Fixture.replay(OpenAIResponsesWire.self, "openai-responses", name)
            .flatMap { $0 }

        let calls = parts.compactMap {
            if case .toolCall(let call) = $0 { return call } else { return nil }
        }
        let serverCalls = calls.filter(\.providerExecuted)

        #expect(!serverCalls.isEmpty, "\(name): no provider-executed call")
        // A shell or apply-patch call mistaken for a client one would run
        // twice, with real side effects.
        #expect(parts.contains { if case .toolResult = $0 { return true } else { return false } })
    }

    @Test("Responses server tool input parses even when it streams bare text")
    func responsesWrapsNonJSONInput() throws {
        // Several tools stream a shell command, a patch diff or a block of
        // code rather than JSON. The contract that assembled input parses has
        // to hold anyway.
        var wire = OpenAIResponsesWire()
        _ = wire.map(chunk: [
            "type": "response.output_item.added",
            "output_index": 0,
            "item": [
                "type": "shell_call", "id": "sh_1", "call_id": "call_sh",
                "status": "in_progress", "action": ["commands": []],
            ],
        ])
        _ = wire.map(chunk: [
            "type": "response.shell_call_command.delta",
            "item_id": "sh_1",
            "output_index": 0,
            "command_index": 0,
            "delta": "ls -la /tmp",
        ])
        let parts = wire.map(chunk: [
            "type": "response.output_item.done",
            "output_index": 0,
            "item": [
                "type": "shell_call", "id": "sh_1", "call_id": "call_sh",
                "status": "completed", "action": ["commands": ["ls -la /tmp"]],
            ],
        ])

        let call = parts.compactMap { if case .toolCall(let c) = $0 { return c } else { return nil } }.first
        #expect(call?.providerExecuted == true)
        #expect((try? JSONValue.decode(from: call?.input ?? "")) != nil)
    }

    @Test("an unknown Responses tool still normalizes")
    func responsesHandlesUnknownTool() {
        // Matching the event shape rather than a list of tool names means a
        // tool OpenAI ships next works without a code change.
        var wire = OpenAIResponsesWire()
        _ = wire.map(chunk: [
            "type": "response.output_item.added",
            "output_index": 0,
            "item": ["type": "future_tool_call", "id": "ft_1"],
        ])
        _ = wire.map(chunk: [
            "type": "response.future_tool_call_thing.delta",
            "item_id": "ft_1",
            "delta": "{\"a\":1}",
        ])
        let parts = wire.map(chunk: [
            "type": "response.output_item.done",
            "output_index": 0,
            "item": ["type": "future_tool_call", "id": "ft_1"],
        ])

        let call = parts.compactMap { if case .toolCall(let c) = $0 { return c } else { return nil } }.first
        #expect(call?.toolName == "future_tool_call")
        #expect(call?.providerExecuted == true)
    }

    @Test("MCP calls are marked dynamic on Responses")
    func responsesMarksMCPDynamic() {
        var wire = OpenAIResponsesWire()
        _ = wire.map(chunk: [
            "type": "response.output_item.added",
            "output_index": 0,
            "item": [
                "type": "mcp_call", "id": "mcp_1", "call_id": "call_mcp",
                "arguments": "", "name": "lookup", "server_label": "server",
            ],
        ])
        let parts = wire.map(chunk: [
            "type": "response.output_item.done",
            "output_index": 0,
            "item": [
                "type": "mcp_call", "id": "mcp_1", "call_id": "call_mcp",
                "arguments": "{}", "name": "lookup", "server_label": "server",
            ],
        ])

        let call = parts.compactMap { if case .toolCall(let c) = $0 { return c } else { return nil } }.first
        #expect(call?.dynamic == true)
        #expect(call?.providerExecuted == true)
    }

    @Test("Gemini code execution becomes a call and a result")
    func geminiNormalizesCodeExecution() {
        var wire = GoogleGenerativeAIWire()
        let parts = wire.map(chunk: [
            "candidates": [[
                "index": 0,
                "content": ["parts": [
                    ["executableCode": ["language": "PYTHON", "code": "print(1)"]],
                    ["codeExecutionResult": ["outcome": "OUTCOME_OK", "output": "1\n"]],
                ]],
            ]],
        ])

        let call = parts.compactMap { if case .toolCall(let c) = $0 { return c } else { return nil } }.first
        #expect(call?.toolName == "code_execution")
        // The caller must not run this code itself.
        #expect(call?.providerExecuted == true)

        let result = parts.compactMap {
            if case .toolResult(let r) = $0 { return r } else { return nil }
        }.first
        // The result carries no id of its own and is paired with the execution
        // that preceded it.
        #expect(result?.toolCallId == call?.toolCallId)
        #expect(result?.isError == false)
    }

    @Test("a failed Gemini execution is flagged as an error")
    func geminiFlagsFailedExecution() {
        var wire = GoogleGenerativeAIWire()
        let parts = wire.map(chunk: [
            "candidates": [[
                "index": 0,
                "content": ["parts": [
                    ["executableCode": ["language": "PYTHON", "code": "1/0"]],
                    ["codeExecutionResult": ["outcome": "OUTCOME_FAILED", "output": "ZeroDivisionError"]],
                ]],
            ]],
        ])

        let result = parts.compactMap {
            if case .toolResult(let r) = $0 { return r } else { return nil }
        }.first
        #expect(result?.isError == true)
    }

    @Test("Gemini grounding becomes sources, deduplicated")
    func geminiNormalizesGrounding() {
        // Grounding metadata repeats on every chunk, so emitting it verbatim
        // would produce one duplicate citation per chunk.
        var wire = GoogleGenerativeAIWire()
        let chunk: JSONValue = [
            "candidates": [[
                "index": 0,
                "content": ["parts": [["text": "hi"]]],
                "groundingMetadata": ["groundingChunks": [
                    ["web": ["uri": "https://swift.org", "title": "Swift"]],
                    ["web": ["uri": "https://apple.com", "title": "Apple"]],
                ]],
            ]],
        ]

        let first = wire.map(chunk: chunk)
        let second = wire.map(chunk: chunk)

        func sources(_ parts: [StreamPart]) -> [Source] {
            parts.compactMap { if case .source(let s) = $0 { return s } else { return nil } }
        }

        #expect(sources(first).count == 2)
        #expect(sources(second).isEmpty, "grounding was re-emitted on a repeat chunk")
    }
}
