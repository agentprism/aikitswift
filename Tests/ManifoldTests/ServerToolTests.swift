import Foundation
import Testing

@testable import Manifold

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
