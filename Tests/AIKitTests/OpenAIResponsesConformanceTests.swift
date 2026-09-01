import Foundation
import Testing

@testable import AIKit

@Suite("OpenAI Responses wire")
struct OpenAIResponsesConformanceTests {

    struct MalformedEventCase: Sendable, CustomTestStringConvertible {
        var name: String
        var event: JSONValue
        var expectedField: String

        var testDescription: String { name }
    }

    struct OutputItemCase: Sendable, CustomTestStringConvertible {
        var item: JSONValue
        var malformedField: String
        var malformedValue: JSONValue

        var testDescription: String { item["type"]?.stringValue ?? "missing-type" }

        var malformedItem: JSONValue {
            guard var object = item.objectValue else { return item }
            object[malformedField] = malformedValue
            return .object(object)
        }
    }

    static let set = "openai-responses"

    static var streamNames: [String] {
        get throws { try Fixture.streamNames(set) }
    }

    /// Independent test oracle from the official generated Responses event
    /// union plus the captured apply-patch events. Do not derive this from the
    /// mapper: doing so would let a missing event disappear from both the
    /// implementation and its tests at the same time.
    static let expectedKnownEventTypes: Set<String> = [
        "error",
        "response.apply_patch_call_operation_diff.delta",
        "response.apply_patch_call_operation_diff.done",
        "response.audio.delta",
        "response.audio.done",
        "response.audio.transcript.delta",
        "response.audio.transcript.done",
        "response.code_interpreter_call.completed",
        "response.code_interpreter_call.in_progress",
        "response.code_interpreter_call.interpreting",
        "response.code_interpreter_call_code.delta",
        "response.code_interpreter_call_code.done",
        "response.completed",
        "response.content_part.added",
        "response.content_part.done",
        "response.created",
        "response.custom_tool_call_input.delta",
        "response.custom_tool_call_input.done",
        "response.failed",
        "response.file_search_call.completed",
        "response.file_search_call.in_progress",
        "response.file_search_call.searching",
        "response.function_call_arguments.delta",
        "response.function_call_arguments.done",
        "response.image_generation_call.completed",
        "response.image_generation_call.generating",
        "response.image_generation_call.in_progress",
        "response.image_generation_call.partial_image",
        "response.in_progress",
        "response.incomplete",
        "response.mcp_call.completed",
        "response.mcp_call.failed",
        "response.mcp_call.in_progress",
        "response.mcp_call_arguments.delta",
        "response.mcp_call_arguments.done",
        "response.mcp_list_tools.completed",
        "response.mcp_list_tools.failed",
        "response.mcp_list_tools.in_progress",
        "response.output_item.added",
        "response.output_item.done",
        "response.output_text.annotation.added",
        "response.output_text.delta",
        "response.output_text.done",
        "response.queued",
        "response.reasoning_summary_part.added",
        "response.reasoning_summary_part.done",
        "response.reasoning_summary_text.delta",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.delta",
        "response.reasoning_text.done",
        "response.refusal.delta",
        "response.refusal.done",
        "response.shell_call_command.added",
        "response.shell_call_command.delta",
        "response.shell_call_command.done",
        "response.shell_call_output_content.delta",
        "response.shell_call_output_content.done",
        "response.web_search_call.completed",
        "response.web_search_call.in_progress",
        "response.web_search_call.searching",
    ]

    static let expectedOutputItemTypes: Set<String> = [
        "message", "file_search_call", "function_call", "function_call_output",
        "web_search_call", "computer_call", "computer_call_output", "reasoning",
        "program", "program_output", "tool_search_call", "tool_search_output",
        "additional_tools", "compaction", "image_generation_call", "code_interpreter_call",
        "local_shell_call", "local_shell_call_output", "shell_call", "shell_call_output",
        "apply_patch_call", "apply_patch_call_output", "mcp_call", "mcp_list_tools",
        "mcp_approval_request", "mcp_approval_response", "custom_tool_call",
        "custom_tool_call_output",
    ]

    /// One independently-authored valid and malformed terminal instance for
    /// every known output item discriminator.
    static let outputItemCases: [OutputItemCase] = [
        .init(
            item: [
                "type": "message", "id": "msg_1", "role": "assistant", "status": "completed",
                "content": [["type": "output_text", "text": "ok", "annotations": []]],
            ],
            malformedField: "role", malformedValue: 7
        ),
        .init(
            item: [
                "type": "file_search_call", "id": "fs_1", "status": "completed",
                "queries": ["needle"], "results": nil,
            ],
            malformedField: "queries", malformedValue: false
        ),
        .init(
            item: [
                "type": "function_call", "id": "fc_1", "call_id": "call_1",
                "name": "lookup", "arguments": "{}", "status": "completed",
            ],
            malformedField: "arguments", malformedValue: [:]
        ),
        .init(
            item: [
                "type": "function_call_output", "id": "fco_1", "call_id": "call_1",
                "output": [["type": "input_text", "text": "done"]], "status": "completed",
            ],
            malformedField: "output", malformedValue: [:]
        ),
        .init(
            item: [
                "type": "web_search_call", "id": "ws_1", "status": "completed",
                "action": ["type": "search", "query": "swift"],
            ],
            malformedField: "action", malformedValue: []
        ),
        .init(
            item: [
                "type": "computer_call", "id": "cc_1", "call_id": "call_2",
                "status": "completed", "pending_safety_checks": [],
                "action": ["type": "screenshot"],
            ],
            malformedField: "pending_safety_checks", malformedValue: [:]
        ),
        .init(
            item: [
                "type": "computer_call_output", "id": "cco_1", "call_id": "call_2",
                "status": "completed", "output": ["type": "computer_screenshot", "image_url": "data:"],
            ],
            malformedField: "output", malformedValue: []
        ),
        .init(
            item: [
                "type": "reasoning", "id": "rs_1",
                "summary": [["type": "summary_text", "text": "why"]],
                "encrypted_content": "opaque",
            ],
            malformedField: "summary", malformedValue: [:]
        ),
        .init(
            item: [
                "type": "program", "id": "cm_1", "call_id": "call_3",
                "code": "text('ok')", "fingerprint": "opaque",
            ],
            malformedField: "fingerprint", malformedValue: 7
        ),
        .init(
            item: [
                "type": "program_output", "id": "cmo_1", "call_id": "call_3",
                "result": "ok", "status": "completed",
            ],
            malformedField: "result", malformedValue: []
        ),
        .init(
            item: [
                "type": "tool_search_call", "id": "tsc_1", "call_id": nil,
                "arguments": ["goal": "weather"], "execution": "server", "status": "completed",
            ],
            malformedField: "arguments", malformedValue: "weather"
        ),
        .init(
            item: [
                "type": "tool_search_output", "id": "tso_1", "call_id": nil,
                "execution": "server", "status": "completed", "tools": [],
            ],
            malformedField: "tools", malformedValue: [:]
        ),
        .init(
            item: ["type": "additional_tools", "id": "at_1", "role": "assistant", "tools": []],
            malformedField: "role", malformedValue: 7
        ),
        .init(
            item: ["type": "compaction", "id": "cmp_1", "encrypted_content": "opaque"],
            malformedField: "encrypted_content", malformedValue: []
        ),
        .init(
            item: ["type": "image_generation_call", "id": "ig_1", "status": "completed"],
            malformedField: "status", malformedValue: 7
        ),
        .init(
            item: [
                "type": "code_interpreter_call", "id": "ci_1", "container_id": "cntr_1",
                "status": "completed", "code": "print(1)", "outputs": [],
            ],
            malformedField: "container_id", malformedValue: []
        ),
        .init(
            item: [
                "type": "local_shell_call", "id": "lsh_1", "call_id": "call_4",
                "status": "completed", "action": ["type": "exec", "command": [], "env": [:]],
            ],
            malformedField: "action", malformedValue: []
        ),
        .init(
            item: ["type": "local_shell_call_output", "id": "lsho_1", "output": "{}"],
            malformedField: "output", malformedValue: []
        ),
        .init(
            item: [
                "type": "shell_call", "id": "sh_1", "call_id": "call_5", "status": "completed",
                "action": ["commands": ["pwd"]],
            ],
            malformedField: "action", malformedValue: []
        ),
        .init(
            item: [
                "type": "shell_call_output", "id": "sho_1", "call_id": "call_5",
                "status": "completed", "output": [],
            ],
            malformedField: "output", malformedValue: [:]
        ),
        .init(
            item: [
                "type": "apply_patch_call", "id": "apc_1", "call_id": "call_6",
                "status": "completed", "operation": ["type": "delete_file", "path": "old.txt"],
            ],
            malformedField: "operation", malformedValue: []
        ),
        .init(
            item: [
                "type": "apply_patch_call_output", "id": "apco_1", "call_id": "call_6",
                "status": "completed", "output": "done",
            ],
            malformedField: "status", malformedValue: []
        ),
        .init(
            item: [
                "type": "mcp_call", "id": "mcp_1", "arguments": "{}", "name": "lookup",
                "server_label": "server", "status": "completed", "output": "ok",
            ],
            malformedField: "arguments", malformedValue: [:]
        ),
        .init(
            item: ["type": "mcp_list_tools", "id": "mcpl_1", "server_label": "server", "tools": []],
            malformedField: "tools", malformedValue: [:]
        ),
        .init(
            item: [
                "type": "mcp_approval_request", "id": "mcpr_1", "arguments": "{}",
                "name": "lookup", "server_label": "server",
            ],
            malformedField: "arguments", malformedValue: [:]
        ),
        .init(
            item: [
                "type": "mcp_approval_response", "id": "mcpa_1",
                "approval_request_id": "mcpr_1", "approve": true,
            ],
            malformedField: "approve", malformedValue: "yes"
        ),
        .init(
            item: [
                "type": "custom_tool_call", "id": "ct_1", "call_id": "call_7",
                "name": "write_sql", "input": "SELECT 1", "status": "completed",
            ],
            malformedField: "input", malformedValue: [:]
        ),
        .init(
            item: [
                "type": "custom_tool_call_output", "id": "cto_1", "call_id": "call_7",
                "output": [["type": "input_text", "text": "done"]], "status": "completed",
            ],
            malformedField: "output", malformedValue: [:]
        ),
    ]

    static let malformedEventFamilies: [MalformedEventCase] = [
        .init(
            name: "error payload",
            event: ["type": "error", "error": ["message": 7]],
            expectedField: "error.message"
        ),
        .init(
            name: "response lifecycle",
            event: ["type": "response.created", "response": "not an object"],
            expectedField: "response"
        ),
        .init(
            name: "completed output item subtype",
            event: [
                "type": "response.output_item.done",
                "output_index": 0,
                "item": [
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": 7,
                ],
            ],
            expectedField: "item.arguments"
        ),
        .init(
            name: "content part",
            event: [
                "type": "response.content_part.done",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "part": ["type": "output_text", "text": 7],
            ],
            expectedField: "part.text"
        ),
        .init(
            name: "output text",
            event: [
                "type": "response.output_text.done",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "text": false,
            ],
            expectedField: "text"
        ),
        .init(
            name: "output annotation",
            event: [
                "type": "response.output_text.annotation.added",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "annotation_index": 0,
                "annotation": ["type": 7],
            ],
            expectedField: "annotation.type"
        ),
        .init(
            name: "reasoning summary part",
            event: [
                "type": "response.reasoning_summary_part.done",
                "item_id": "rs_1",
                "output_index": 0,
                "summary_index": 0,
                "part": ["type": "summary_text", "text": 7],
            ],
            expectedField: "part.text"
        ),
        .init(
            name: "reasoning summary text",
            event: [
                "type": "response.reasoning_summary_text.done",
                "item_id": "rs_1",
                "output_index": 0,
                "summary_index": "zero",
                "text": "done",
            ],
            expectedField: "summary_index"
        ),
        .init(
            name: "function arguments",
            event: [
                "type": "response.function_call_arguments.done",
                "item_id": "fc_1",
                "output_index": 0,
                "arguments": [:],
            ],
            expectedField: "arguments"
        ),
        .init(
            name: "apply patch diff",
            event: [
                "type": "response.apply_patch_call_operation_diff.done",
                "item_id": "apc_1",
                "output_index": 0,
                "diff": [],
            ],
            expectedField: "diff"
        ),
        .init(
            name: "code interpreter",
            event: [
                "type": "response.code_interpreter_call_code.done",
                "item_id": "ci_1",
                "output_index": 0,
                "code": false,
            ],
            expectedField: "code"
        ),
        .init(
            name: "custom tool input",
            event: [
                "type": "response.custom_tool_call_input.delta",
                "item_id": "ct_1",
                "output_index": 0,
                "delta": 7,
            ],
            expectedField: "delta"
        ),
        .init(
            name: "tool lifecycle",
            event: [
                "type": "response.web_search_call.completed",
                "item_id": "ws_1",
                "output_index": "zero",
            ],
            expectedField: "output_index"
        ),
        .init(
            name: "partial image",
            event: [
                "type": "response.image_generation_call.partial_image",
                "item_id": "ig_1",
                "output_index": 0,
                "partial_image_index": "zero",
                "partial_image_b64": "aW1hZ2U=",
            ],
            expectedField: "partial_image_index"
        ),
        .init(
            name: "MCP arguments",
            event: [
                "type": "response.mcp_call_arguments.done",
                "item_id": "mcp_1",
                "output_index": 0,
                "arguments": [],
            ],
            expectedField: "arguments"
        ),
        .init(
            name: "shell command",
            event: [
                "type": "response.shell_call_command.done",
                "output_index": 0,
                "command_index": "zero",
                "command": "pwd",
            ],
            expectedField: "command_index"
        ),
        .init(
            name: "shell output",
            event: [
                "type": "response.shell_call_output_content.done",
                "item_id": "sho_1",
                "output_index": 0,
                "command_index": 0,
                "output": [:],
            ],
            expectedField: "output"
        ),
        .init(
            name: "terminal status agreement",
            event: [
                "type": "response.completed",
                "response": [
                    "id": "resp_1",
                    "status": "failed",
                    "output": [],
                ],
            ],
            expectedField: "response.status"
        ),
        .init(
            name: "failed response details",
            event: [
                "type": "response.failed",
                "response": [
                    "id": "resp_1",
                    "status": "failed",
                    "output": [],
                ],
            ],
            expectedField: "response.error"
        ),
    ]

    @Test("fixtures are present")
    func fixturesArePresent() throws {
        #expect(try Self.streamNames.count >= 20)
    }

    @Test("known event catalog matches the independent provider oracle")
    func eventCatalogIsComplete() {
        #expect(OpenAIResponsesWire.knownEventTypes == Self.expectedKnownEventTypes)
    }

    @Test("known output item catalog matches the independent provider oracle")
    func outputItemCatalogIsComplete() {
        #expect(OpenAIResponsesWire.knownOutputItemTypes == Self.expectedOutputItemTypes)
        #expect(Set(Self.outputItemCases.compactMap { $0.item["type"]?.stringValue }) == Self.expectedOutputItemTypes)
    }

    @Test("every known terminal output item subtype validates", arguments: outputItemCases)
    func validatesKnownTerminalOutputItem(testCase: OutputItemCase) {
        let event: JSONValue = [
            "type": "response.completed",
            "response": ["id": "resp_1", "status": "completed", "output": [testCase.item]],
        ]
        var wire = OpenAIResponsesWire()
        let parts = wire.map(chunk: event)

        #expect(!parts.contains {
            if case .error(let error) = $0 { return error.type == "malformed_event" }
            return false
        })
    }

    @Test("every known terminal output item subtype rejects malformed fields", arguments: outputItemCases)
    func rejectsMalformedKnownTerminalOutputItem(testCase: OutputItemCase) {
        let event: JSONValue = [
            "type": "response.completed",
            "response": ["id": "resp_1", "status": "completed", "output": [testCase.malformedItem]],
        ]
        var wire = OpenAIResponsesWire()
        let parts = wire.map(chunk: event)
        let error = parts.compactMap {
            if case .error(let error) = $0, error.type == "malformed_event" { return error }
            return nil
        }.first

        #expect(error?.message.contains("response.output[0].\(testCase.malformedField)") == true)
        #expect(error?.raw == event)
    }

    @Test("recorded streams are well-formed", arguments: try streamNames)
    func conforms(name: String) throws {
        WireConformance.check(
            try Fixture.replay(OpenAIResponsesWire.self, Self.set, name),
            label: name
        )
    }

    @Test("recorded known events retain their complete payload", arguments: try streamNames)
    func preservesRecordedEventPayloads(name: String) throws {
        var wire = OpenAIResponsesWire()
        for chunk in try Fixture.chunks(Self.set, name) {
            let type = try #require(chunk["type"]?.stringValue)
            #expect(Self.expectedKnownEventTypes.contains(type), "\(name): fixture introduced uncataloged event \(type)")
            let parts = wire.map(chunk: chunk)
            #expect(parts.contains {
                if case .providerEvent(let event) = $0 {
                    return event.type == type && event.payload == chunk
                }
                return false
            }, "\(name): known event \(type) was not retained")
            #expect(!parts.contains {
                if case .error(let error) = $0 { return error.type == "malformed_event" }
                return false
            }, "\(name): valid recorded event \(type) was treated as malformed")
        }
    }

    @Test("statuses map to the normalized set")
    func mapsStatuses() {
        #expect(OpenAIResponsesWire.mapStatus("completed").unified == .stop)
        // The API reports success without saying tools are pending, so the
        // presence of a call is the only signal the caller gets.
        #expect(OpenAIResponsesWire.mapStatus("completed", hasToolCalls: true).unified == .toolCalls)
        #expect(
            OpenAIResponsesWire.mapStatus("incomplete", incompleteReason: "max_output_tokens").unified == .length
        )
        #expect(
            OpenAIResponsesWire.mapStatus("incomplete", incompleteReason: "content_filter").unified == .contentFilter
        )
        #expect(OpenAIResponsesWire.mapStatus("failed").unified == .error)

        let future = OpenAIResponsesWire.mapStatus("something_new")
        #expect(future.unified == .other)
        #expect(future.raw == "something_new")
    }

    @Test("cached and reasoning tokens are subtracted, not added")
    func treatsDetailsAsIncluded() {
        // Both totals are inclusive here, matching Chat Completions and
        // differing from Anthropic (input) and Gemini (output).
        let usage = OpenAIResponsesWire.convertUsage([
            "input_tokens": 1000,
            "output_tokens": 500,
            "input_tokens_details": ["cached_tokens": 900],
            "output_tokens_details": ["reasoning_tokens": 400],
        ])

        #expect(usage.inputTokens.total == 1000)
        #expect(usage.inputTokens.noCache == 100)
        #expect(usage.inputTokens.cacheRead == 900)
        #expect(usage.outputTokens.total == 500)
        #expect(usage.outputTokens.text == 100)
        #expect(usage.outputTokens.reasoning == 400)
    }

    @Test("tool results reference call_id, not the streaming item id")
    func usesCallIdForToolCalls() {
        // `id` addresses the streaming item; `call_id` is what a tool result
        // must reference. Confusing them produces a result the model cannot
        // match back to its call.
        var wire = OpenAIResponsesWire()

        _ = wire.map(chunk: [
            "type": "response.output_item.added",
            "output_index": 0,
            "item": [
                "type": "function_call",
                "id": "item_1",
                "call_id": "call_xyz",
                "name": "get_weather",
                "arguments": "",
            ],
        ])
        _ = wire.map(chunk: [
            "type": "response.function_call_arguments.delta",
            "item_id": "item_1",
            "output_index": 0,
            "delta": "{\"city\":\"Paris\"}",
        ])
        let parts = wire.map(chunk: [
            "type": "response.output_item.done",
            "output_index": 0,
            "item": [
                "type": "function_call",
                "id": "item_1",
                "call_id": "call_xyz",
                "name": "get_weather",
                "arguments": "{\"city\":\"Paris\"}",
            ],
        ])

        let call = parts.compactMap { if case .toolCall(let c) = $0 { return c } else { return nil } }.first
        #expect(call?.toolCallId == "call_xyz")
        #expect(call?.toolName == "get_weather")
        #expect(call?.input == "{\"city\":\"Paris\"}")
    }

    @Test("text blocks are keyed by item and content index")
    func keysTextByItemAndContentIndex() {
        // A single item can carry several content parts. Keying on item alone
        // would collapse them and unbalance the triads.
        var wire = OpenAIResponsesWire()

        for index in 0..<2 {
            _ = wire.map(chunk: [
                "type": "response.content_part.added",
                "item_id": "item_1",
                "output_index": 0,
                "content_index": .number(Double(index)),
                "part": ["type": "output_text", "text": "", "annotations": []],
            ])
        }

        let starts = wire.map(chunk: [
            "type": "response.content_part.done",
            "item_id": "item_1",
            "output_index": 0,
            "content_index": 0,
            "part": ["type": "output_text", "text": "", "annotations": []],
        ])

        #expect(starts.contains {
            if case .textEnd(let id, _) = $0 { return id == "item_1#0" } else { return false }
        })
    }

    @Test("encrypted reasoning and output item identity replay exactly")
    func replaysEncryptedReasoning() throws {
        let name = "openai-reasoning-encrypted-content.1"
        let chunks = try Fixture.chunks(Self.set, name)
        let terminalIndex = try #require(chunks.firstIndex { chunk in
            chunk["type"]?.stringValue == "response.completed"
                && (chunk["response"]?["output"]?.arrayValue ?? []).contains {
                    $0["type"]?.stringValue == "reasoning"
                }
        })
        let terminal = chunks[terminalIndex]
        let expected = try #require(terminal["response"]?["output"]?.arrayValue)
        let startIndex = try #require(chunks[...terminalIndex].lastIndex {
            $0["type"]?.stringValue == "response.created"
        })
        var wire = OpenAIResponsesWire()
        var parts = chunks[startIndex...terminalIndex].flatMap { wire.map(chunk: $0) }
        parts += wire.finish()

        var prompt: Prompt = [.user("calculate")]
        prompt.append(AIResponse(parts: parts).assistantMessage)
        let input = OpenAIResponsesRequest.encode(
            CallOptions(model: "gpt-5.1-codex-max", prompt: prompt)
        ).body["input"]?.arrayValue
        let replayed = input.map { Array($0.suffix(expected.count)) }

        #expect(replayed == expected)
        let reasoning = try #require(replayed?.first { $0["type"]?.stringValue == "reasoning" })
        #expect(reasoning["id"] == expected.first?["id"])
        #expect(reasoning["encrypted_content"] == expected.first?["encrypted_content"])
    }

    @Test("assistant message id, status, and phase replay exactly")
    func replaysMessageIdentityAndPhase() throws {
        let name = "openai-phase.1"
        let chunks = try Fixture.chunks(Self.set, name)
        let terminal = try #require(chunks.last { $0["type"]?.stringValue == "response.completed" })
        let expected = (terminal["response"]?["output"]?.arrayValue ?? [])
            .filter { $0["type"]?.stringValue == "message" }
        let parts = try #require(try Fixture.replay(OpenAIResponsesWire.self, Self.set, name).first)
        let message = AIResponse(parts: parts).assistantMessage
        let replayed = OpenAIResponsesRequest.encode(
            CallOptions(model: "gpt-5.3-codex", prompt: [.user("news"), message])
        ).body["input"]?.arrayValue?.filter { $0["type"]?.stringValue == "message" }

        #expect(replayed == expected)
        #expect(replayed?.map { $0["id"] } == expected.map { $0["id"] })
        #expect(replayed?.map { $0["status"] } == expected.map { $0["status"] })
        #expect(replayed?.map { $0["phase"] } == expected.map { $0["phase"] })
    }

    @Test("function item id, call_id, and namespace remain distinct")
    func preservesFunctionIdentityAndNamespace() throws {
        let parts = try #require(
            try Fixture.replay(OpenAIResponsesWire.self, Self.set, "openai-tool-search.1").first
        )
        let response = AIResponse(parts: parts)
        let call = try #require(response.toolCalls.first { $0.toolName == "get_weather" })

        #expect(call.toolCallId == "call_pddfxhfOx4gY56zn4vIIEbFp")
        #expect(call.namespace == "get_weather")
        #expect(call.providerMetadata?["openai"]?["item"]?["id"]?.stringValue?.hasPrefix("fc_") == true)
        #expect(call.providerMetadata?["openai"]?["item"]?["id"] != .string(call.toolCallId))

        let input = OpenAIResponsesRequest.encode(CallOptions(
            model: "gpt-5.4", prompt: [.user("weather"), response.assistantMessage]
        )).body["input"]?.arrayValue
        let replayed = try #require(input?.first { $0["type"]?.stringValue == "function_call" })
        #expect(replayed["id"] == call.providerMetadata?["openai"]?["item"]?["id"])
        #expect(replayed["call_id"]?.stringValue == call.toolCallId)
        #expect(replayed["namespace"]?.stringValue == call.namespace)
    }

    @Test("custom tool calls stay pending and replay exact grammar input")
    func preservesCustomToolCallSemantics() throws {
        let parts = try #require(
            try Fixture.replay(OpenAIResponsesWire.self, Self.set, "openai-custom-tool.1").first
        )
        let response = AIResponse(parts: parts)
        let call = try #require(response.pendingToolCalls.first)
        let terminal = try #require(response.providerEvents.last {
            $0.type == "response.completed"
        })
        let expectedCallItem = try #require(terminal.payload["response"]?["output"]?.arrayValue?.first)

        #expect(call.toolCallId == "call_custom_sql_001")
        #expect(call.toolName == "write_sql")
        #expect(call.input == "SELECT * FROM users WHERE age > 25")
        #expect(call.providerExecuted == false)
        #expect(response.toolResults.isEmpty)

        let result = Message.toolResult(
            toolCallId: call.toolCallId,
            toolName: call.toolName,
            result: "accepted"
        )
        let input = OpenAIResponsesRequest.encode(CallOptions(
            model: "gpt-5.4", prompt: [.user("query"), response.assistantMessage, result]
        )).body["input"]?.arrayValue

        #expect(input?.suffix(2).first == expectedCallItem)
        #expect(input?.last == .object([
            "type": "custom_tool_call_output",
            "call_id": .string(call.toolCallId),
            "output": "accepted",
        ]))
    }

    @Test("known events preserve their payload and unknown events remain raw")
    func preservesKnownAndUnknownEvents() {
        let known: JSONValue = [
            "type": "response.in_progress",
            "sequence_number": 7,
            "response": ["id": "resp_1", "status": "in_progress"],
        ]
        let unknown: JSONValue = ["type": "response.future_event", "value": 42]
        var wire = OpenAIResponsesWire()

        let knownParts = wire.map(chunk: known)
        #expect(knownParts.contains {
            if case .providerEvent(let event) = $0 { return event.payload == known }
            return false
        })
        #expect(!knownParts.contains { if case .raw = $0 { return true } else { return false } })

        let unknownParts = wire.map(chunk: unknown)
        #expect(unknownParts.contains { if case .raw(let payload) = $0 { return payload == unknown } else { return false } })
    }

    @Test("malformed known events surface errors")
    func rejectsMalformedKnownEvent() {
        var wire = OpenAIResponsesWire()
        let parts = wire.map(chunk: [
            "type": "response.output_text.delta",
            "item_id": "msg_1",
            "content_index": 0,
        ])

        let error = parts.compactMap { if case .error(let error) = $0 { return error } else { return nil } }.first
        #expect(error?.type == "malformed_event")
        #expect(error?.raw?["type"]?.stringValue == "response.output_text.delta")
    }

    @Test(
        "every independently cataloged event type rejects an absent payload",
        arguments: expectedKnownEventTypes.sorted()
    )
    func validatesEveryKnownEventType(type: String) {
        let event: JSONValue = ["type": .string(type)]
        var wire = OpenAIResponsesWire()
        let parts = wire.map(chunk: event)

        #expect(parts.contains {
            if case .providerEvent(let providerEvent) = $0 {
                return providerEvent.type == type && providerEvent.payload == event
            }
            return false
        })
        #expect(parts.contains {
            if case .error(let error) = $0 {
                return error.type == "malformed_event" && error.raw == event
            }
            return false
        }, "\(type) bypassed known-event validation")
    }

    @Test("malformed known event families retain their payload", arguments: malformedEventFamilies)
    func rejectsMalformedEventFamily(testCase: MalformedEventCase) {
        var wire = OpenAIResponsesWire()
        let parts = wire.map(chunk: testCase.event)
        let error = parts.compactMap {
            if case .error(let error) = $0 { return error }
            return nil
        }.first

        #expect(error?.type == "malformed_event")
        #expect(error?.message.contains(testCase.expectedField) == true)
        #expect(error?.raw == testCase.event)
        #expect(parts.contains {
            if case .providerEvent(let event) = $0 { return event.payload == testCase.event }
            return false
        })
    }

    @Test("terminal response and nested provider errors are retained")
    func preservesTerminalFailureDetails() throws {
        let parts = try #require(
            try Fixture.replay(OpenAIResponsesWire.self, Self.set, "openai-error.1").first
        )
        let response = AIResponse(parts: parts)

        #expect(response.errors.contains { $0.type == "insufficient_quota" && $0.message.contains("quota") })
        #expect(response.providerEvents.contains { $0.type == "response.failed" })
        guard case .finish(_, let reason, let metadata)? = parts.last else {
            Issue.record("expected terminal finish")
            return
        }
        #expect(reason.unified == .error)
        #expect(metadata?["openai"]?["response"]?["error"]?["code"]?.stringValue == "insufficient_quota")
    }

    @Test("documented top-level provider errors are retained")
    func preservesTopLevelProviderError() {
        let event: JSONValue = [
            "type": "error",
            "code": "rate_limit_exceeded",
            "message": "Slow down",
            "param": nil,
            "sequence_number": 17,
        ]
        var wire = OpenAIResponsesWire()
        let parts = wire.map(chunk: event)

        #expect(parts.contains {
            if case .providerEvent(let providerEvent) = $0 { return providerEvent.payload == event }
            return false
        })
        #expect(parts.contains {
            if case .error(let error) = $0 {
                return error.type == "rate_limit_exceeded"
                    && error.message == "Slow down"
                    && error.raw == event
            }
            return false
        })
        #expect(!parts.contains {
            if case .error(let error) = $0 { return error.type == "malformed_event" }
            return false
        })
    }

    @Test("documented top-level provider errors allow an absent error code")
    func preservesTopLevelProviderErrorWithoutCode() {
        let event: JSONValue = [
            "type": "error",
            "code": nil,
            "message": "The request could not be completed",
            "param": nil,
            "sequence_number": 18,
        ]
        var wire = OpenAIResponsesWire()
        let parts = wire.map(chunk: event)

        #expect(parts.contains {
            if case .providerEvent(let providerEvent) = $0 { return providerEvent.payload == event }
            return false
        })
        #expect(parts.contains {
            if case .error(let error) = $0 {
                return error.type == nil
                    && error.message == "The request could not be completed"
                    && error.raw == event
            }
            return false
        })
        #expect(!parts.contains {
            if case .error(let error) = $0 { return error.type == "malformed_event" }
            return false
        })
    }
}
