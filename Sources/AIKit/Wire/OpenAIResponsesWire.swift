import Foundation

/// Maps the OpenAI Responses API streaming protocol onto ``StreamPart``.
///
/// Structurally the opposite of Chat Completions. Where Completions sends one
/// chunk shape and leaves the caller to infer structure from deltas, Responses
/// sends explicit lifecycle events (`output_item.added`, `content_part.added`,
/// `…done`) that already describe the block structure. Less inference, many
/// more event types.
///
/// Every known provider event is retained as a ``ProviderEvent``. Events with
/// a provider-neutral equivalent additionally emit normalized parts; unknown
/// future event types remain available through ``StreamPart/raw(_:)``.
public struct OpenAIResponsesWire: WireMapper {

    private struct PendingToolCall {
        var callId: String
        var name: String
        var namespace: String?
        var arguments: String
        var itemType: String
    }

    /// Function calls keyed by the item id their argument deltas reference.
    private var pendingToolCalls: [String: PendingToolCall] = [:]
    private var toolCallOrder: [String] = []
    /// Provider-executed items — web search, code interpreter, shell, MCP,
    /// apply patch — keyed the same way.
    private var pendingServerTools: [String: PendingToolCall] = [:]
    private var pendingReasoning: [String: JSONValue] = [:]

    /// Item types the protocol defines itself, rather than provider-executed
    /// tool activity.
    private static let ownItemTypes: Set<String> = [
        "message", "reasoning", "function_call", "custom_tool_call", "compaction",
    ]

    /// Provider-tool events whose `delta` is call input. Other delta events can
    /// carry structured output (for example shell stdout) and must not be
    /// coerced into a string argument stream.
    private static let serverToolInputDeltaEvents: Set<String> = [
        "response.apply_patch_call_operation_diff.delta",
        "response.code_interpreter_call_code.delta",
        "response.mcp_call_arguments.delta",
        "response.shell_call_command.delta",
    ]

    private enum JSONKind: String {
        case string
        case integer
        case object
        case array
        case boolean

        func accepts(_ value: JSONValue) -> Bool {
            switch self {
            case .string: value.stringValue != nil
            case .integer: value.intValue != nil
            case .object: value.objectValue != nil
            case .array: value.arrayValue != nil
            case .boolean: value.boolValue != nil
            }
        }
    }

    private struct FieldRule {
        var path: [String]
        var kind: JSONKind
        var required: Bool

        init(_ path: String, _ kind: JSONKind, required: Bool = true) {
            self.path = path.split(separator: ".").map(String.init)
            self.kind = kind
            self.required = required
        }

        var name: String { path.joined(separator: ".") }
    }

    /// Required wire shapes for every event the mapper recognizes. Deriving the
    /// known-type set from this table makes it impossible to add a recognized
    /// event that silently bypasses validation.
    private static let eventSchemas: [String: [FieldRule]] = {
        var schemas: [String: [FieldRule]] = [:]
        // Every event in the official streaming union carries a required
        // sequence number. Treating it as optional allowed truncated or
        // hand-shaped known events to pass validation.
        let sequence = FieldRule("sequence_number", .integer)
        let itemReference = [
            sequence,
            FieldRule("item_id", .string),
            FieldRule("output_index", .integer),
        ]

        func register(_ types: [String], _ fields: [FieldRule]) {
            for type in types { schemas[type] = fields }
        }

        // Error events have two provider shapes: the documented top-level
        // `code`/`message` fields and an older captured nested `error` object.
        // Their alternative requirements are checked by `validateProviderError`.
        register(["error"], [sequence])
        register(["response.created", "response.in_progress", "response.queued"], [
            sequence,
            FieldRule("response", .object),
            FieldRule("response.id", .string),
            FieldRule("response.status", .string),
        ])
        register(["response.output_item.added", "response.output_item.done"], [
            sequence,
            FieldRule("output_index", .integer),
            FieldRule("item", .object),
            FieldRule("item.type", .string),
        ])
        register(["response.content_part.added", "response.content_part.done"], itemReference + [
            FieldRule("content_index", .integer),
            FieldRule("part", .object),
            FieldRule("part.type", .string),
        ])
        register(["response.output_text.delta"], itemReference + [
            FieldRule("content_index", .integer),
            FieldRule("delta", .string),
            FieldRule("logprobs", .array, required: false),
        ])
        register(["response.output_text.done"], itemReference + [
            FieldRule("content_index", .integer),
            FieldRule("text", .string),
            FieldRule("logprobs", .array, required: false),
        ])
        register(["response.refusal.delta"], itemReference + [
            FieldRule("content_index", .integer),
            FieldRule("delta", .string),
        ])
        register(["response.refusal.done"], itemReference + [
            FieldRule("content_index", .integer),
            FieldRule("refusal", .string),
        ])
        register(["response.output_text.annotation.added"], itemReference + [
            FieldRule("content_index", .integer),
            FieldRule("annotation_index", .integer),
            FieldRule("annotation", .object),
            FieldRule("annotation.type", .string),
        ])
        register([
            "response.reasoning_summary_part.added",
            "response.reasoning_summary_part.done",
        ], itemReference + [
            FieldRule("summary_index", .integer),
            FieldRule("part", .object),
            FieldRule("part.type", .string),
            FieldRule("part.text", .string),
        ])
        register(["response.reasoning_summary_text.delta"], itemReference + [
            FieldRule("summary_index", .integer),
            FieldRule("delta", .string),
        ])
        register(["response.reasoning_summary_text.done"], itemReference + [
            FieldRule("summary_index", .integer),
            FieldRule("text", .string),
        ])
        register(["response.reasoning_text.delta"], itemReference + [
            FieldRule("content_index", .integer),
            FieldRule("delta", .string),
        ])
        register(["response.reasoning_text.done"], itemReference + [
            FieldRule("content_index", .integer),
            FieldRule("text", .string),
        ])
        register(["response.function_call_arguments.delta"], itemReference + [
            FieldRule("delta", .string),
        ])
        register(["response.function_call_arguments.done"], itemReference + [
            FieldRule("arguments", .string),
            FieldRule("name", .string),
        ])
        register(["response.apply_patch_call_operation_diff.delta"], itemReference + [
            FieldRule("delta", .string),
        ])
        register(["response.apply_patch_call_operation_diff.done"], itemReference + [
            FieldRule("diff", .string),
        ])
        register([
            "response.code_interpreter_call.in_progress",
            "response.code_interpreter_call.interpreting",
            "response.code_interpreter_call.completed",
            "response.file_search_call.in_progress",
            "response.file_search_call.searching",
            "response.file_search_call.completed",
            "response.image_generation_call.in_progress",
            "response.image_generation_call.generating",
            "response.image_generation_call.completed",
            "response.mcp_call.in_progress",
            "response.mcp_call.completed",
            "response.mcp_call.failed",
            "response.mcp_list_tools.in_progress",
            "response.mcp_list_tools.completed",
            "response.mcp_list_tools.failed",
            "response.web_search_call.in_progress",
            "response.web_search_call.searching",
            "response.web_search_call.completed",
        ], itemReference)
        register(["response.code_interpreter_call_code.delta"], itemReference + [
            FieldRule("delta", .string),
        ])
        register(["response.code_interpreter_call_code.done"], itemReference + [
            FieldRule("code", .string),
        ])
        register(["response.custom_tool_call_input.delta"], itemReference + [
            FieldRule("delta", .string),
        ])
        register(["response.custom_tool_call_input.done"], itemReference + [
            FieldRule("input", .string),
        ])
        register(["response.image_generation_call.partial_image"], itemReference + [
            FieldRule("partial_image_index", .integer),
            FieldRule("partial_image_b64", .string),
            FieldRule("size", .string, required: false),
            FieldRule("quality", .string, required: false),
            FieldRule("background", .string, required: false),
            FieldRule("output_format", .string, required: false),
        ])
        register(["response.mcp_call_arguments.delta"], itemReference + [
            FieldRule("delta", .string),
        ])
        register(["response.mcp_call_arguments.done"], itemReference + [
            FieldRule("arguments", .string),
        ])
        register(["response.shell_call_command.added", "response.shell_call_command.done"], [
            sequence,
            FieldRule("output_index", .integer),
            FieldRule("command_index", .integer),
            FieldRule("command", .string),
        ])
        register(["response.shell_call_command.delta"], [
            sequence,
            FieldRule("output_index", .integer),
            FieldRule("command_index", .integer),
            FieldRule("delta", .string),
        ])
        register(["response.shell_call_output_content.delta"], itemReference + [
            FieldRule("command_index", .integer),
            FieldRule("delta", .object),
        ])
        register(["response.shell_call_output_content.done"], itemReference + [
            FieldRule("command_index", .integer),
            FieldRule("output", .array),
        ])
        register(["response.completed", "response.incomplete", "response.failed"], [
            sequence,
            FieldRule("response", .object),
            FieldRule("response.id", .string),
            FieldRule("response.status", .string),
            FieldRule("response.output", .array),
        ])
        // These documented audio events are absent from the older fixture
        // corpus. Their sequence number is required because it is the only
        // event-specific payload on the completion events.
        register(["response.audio.delta"], [
            FieldRule("sequence_number", .integer),
            FieldRule("delta", .string),
        ])
        register(["response.audio.done", "response.audio.transcript.done"], [
            FieldRule("sequence_number", .integer),
        ])
        register(["response.audio.transcript.delta"], [
            FieldRule("sequence_number", .integer),
            FieldRule("delta", .string),
        ])

        return schemas
    }()

    /// Provider events covered by this mapper. Internal visibility lets the
    /// conformance suite prove that every recognized type has a malformed case.
    static let knownEventTypes: Set<String> = Set(eventSchemas.keys)

    private var textOpen: Set<String> = []
    private var reasoningOpen: Set<String> = []

    private var finishReason: FinishReason = .other
    private var emittedClientToolCall = false
    private var usage: JSONValue?
    private var terminalResponse: JSONValue?
    private var emittedStreamStart = false
    private var finished = false

    public init() {}

    // MARK: - Entry points

    public mutating func map(chunk: JSONValue) -> [StreamPart] {
        var parts: [StreamPart] = []

        if !emittedStreamStart {
            emittedStreamStart = true
            parts.append(.streamStart(warnings: []))
        }

        parts.append(contentsOf: mapChunk(chunk))
        return parts
    }

    /// Maps a Codex terminal alias through the Responses state machine without
    /// imposing the ordinary event-type/status equality rule. Codex uses
    /// `response.done`, `response.completed`, and `response.incomplete` as
    /// terminal aliases whose response status can be any member of its six
    /// value status union. An actual `response.failed` event remains on the
    /// ordinary path and is the only terminal event that emits a provider error.
    mutating func mapTerminalAlias(
        _ chunk: JSONValue,
        allowedStatuses: Set<String>
    ) -> [StreamPart] {
        var parts: [StreamPart] = []
        if !emittedStreamStart {
            emittedStreamStart = true
            parts.append(.streamStart(warnings: []))
        }

        let type = chunk["type"]?.stringValue ?? "response terminal"
        let preserved = StreamPart.providerEvent(ProviderEvent(
            provider: "openai", type: type, payload: chunk
        ))
        parts.append(preserved)

        if let detail = Self.codexTerminalAliasValidationError(
            chunk,
            allowedStatuses: allowedStatuses
        ) {
            parts.append(Self.malformed(type, detail, chunk))
            return parts
        }

        parts.append(contentsOf: responseTerminal(chunk, emitFailureError: false))
        return parts
    }

    public mutating func map(rawJSON: String) -> [StreamPart] {
        let trimmed = rawJSON.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed != "[DONE]" else { return finish() }

        do {
            return map(chunk: try JSONValue.decode(from: trimmed))
        } catch {
            return [.error(StreamError(
                type: "parse_error",
                message: "Failed to decode chunk: \(error)",
                raw: .string(rawJSON)
            ))]
        }
    }

    public mutating func finish() -> [StreamPart] {
        guard !finished else { return [] }
        finished = true

        var parts: [StreamPart] = []

        if !emittedStreamStart {
            emittedStreamStart = true
            parts.insert(.streamStart(warnings: []), at: 0)
        }

        parts.append(contentsOf: closeOpenBlocks())
        parts.append(contentsOf: flushPendingToolCalls())
        parts.append(.finish(
            usage: usage.map(Self.convertUsage) ?? .empty,
            finishReason: finishReason,
            providerMetadata: finishMetadata
        ))

        return parts
    }

    // MARK: - Event dispatch

    private mutating func mapChunk(_ chunk: JSONValue) -> [StreamPart] {
        guard let type = chunk["type"]?.stringValue else {
            return [Self.malformed("event", "missing string field `type`", chunk)]
        }
        guard Self.knownEventTypes.contains(type) else { return [.raw(chunk)] }

        let preserved = StreamPart.providerEvent(ProviderEvent(
            provider: "openai", type: type, payload: chunk
        ))
        if let detail = Self.validationError(for: type, chunk: chunk) {
            return [preserved, Self.malformed(type, detail, chunk)]
        }
        let mapped: [StreamPart]
        switch type {
        case "response.created":
            guard let response = chunk["response"]?.objectValue.map(JSONValue.object) else {
                return [preserved, Self.malformed(type, "missing object field `response`", chunk)]
            }
            mapped = [.responseMetadata(ResponseMetadata(
                id: response["id"]?.stringValue,
                timestamp: response["created_at"]?.doubleValue.map { Date(timeIntervalSince1970: $0) },
                modelId: response["model"]?.stringValue
            ))]

        case "response.in_progress", "response.queued":
            guard chunk["response"]?.objectValue != nil else {
                return [preserved, Self.malformed(type, "missing object field `response`", chunk)]
            }
            mapped = []

        case "response.output_item.added":
            mapped = outputItemAdded(chunk)

        case "response.output_item.done":
            mapped = outputItemDone(chunk)

        case "response.content_part.added":
            mapped = contentPartAdded(chunk)

        case "response.output_text.delta":
            guard let id = textId(chunk), let delta = chunk["delta"]?.stringValue else {
                return [preserved, Self.malformed(type, "missing item_id, content_index, or string delta", chunk)]
            }
            mapped = [.textDelta(id: id, delta: delta)]

        case "response.content_part.done":
            guard let id = textId(chunk) else {
                return [preserved, Self.malformed(type, "missing string field `item_id`", chunk)]
            }
            guard textOpen.remove(id) != nil else { return [preserved] }
            mapped = [.textEnd(id: id)]

        case "response.output_text.done":
            // Redundant with `content_part.done`, which is what closes the
            // block. Emitting on both would end the triad twice.
            guard textId(chunk) != nil, chunk["text"]?.stringValue != nil else {
                return [preserved, Self.malformed(type, "missing item_id or string text", chunk)]
            }
            mapped = []

        case "response.reasoning_summary_text.delta":
            mapped = reasoningDelta(chunk)

        case "response.function_call_arguments.delta":
            mapped = functionArgumentsDelta(chunk)

        case "response.function_call_arguments.done":
            mapped = functionArgumentsDone(chunk)

        case "response.custom_tool_call_input.delta":
            mapped = clientToolInputDelta(chunk)

        case "response.custom_tool_call_input.done":
            mapped = customToolInputDone(chunk)

        case "response.completed", "response.incomplete", "response.failed":
            mapped = responseTerminal(chunk)

        case "error":
            mapped = Self.providerError(chunk)

        case let knownType where Self.serverToolInputDeltaEvents.contains(knownType):
            // Every provider-executed tool streams its input through an event
            // of the form `response.<tool>.delta` carrying `item_id` and
            // `delta`. Matching the shape rather than enumerating tool names
            // means a tool OpenAI ships next works without a code change.
            mapped = serverToolDelta(chunk)

        default:
            // A recognized event with no provider-neutral projection remains
            // available through `providerEvent` above.
            mapped = []
        }

        return [preserved] + mapped
    }

    // MARK: - Items

    private mutating func outputItemAdded(_ chunk: JSONValue) -> [StreamPart] {
        guard let item = chunk["item"]?.objectValue.map(JSONValue.object) else {
            return [Self.malformed("response.output_item.added", "missing object field `item`", chunk)]
        }
        guard let itemKey = itemKey(chunk, item: item) else {
            return [Self.malformed(
                "response.output_item.added", "item needs `id` or event needs `output_index`", chunk
            )]
        }
        guard let itemType = item["type"]?.stringValue else {
            return [Self.malformed("response.output_item.added", "item is missing string field `type`", chunk)]
        }

        switch itemType {
        case "function_call", "custom_tool_call":
            // `call_id` is the identifier the tool result must reference; `id`
            // only addresses the streaming item. Confusing the two produces a
            // tool result the model cannot match to its call.
            guard let callId = item["call_id"]?.stringValue,
                  let name = item["name"]?.stringValue else {
                return [Self.malformed(
                    "response.output_item.added", "function_call needs string call_id and name", chunk
                )]
            }

            let inputField = itemType == "custom_tool_call" ? "input" : "arguments"
            pendingToolCalls[itemKey] = PendingToolCall(
                callId: callId,
                name: name,
                namespace: item["namespace"]?.stringValue,
                arguments: item[inputField]?.stringValue ?? "",
                itemType: itemType
            )
            toolCallOrder.append(itemKey)
            return [.toolInputStart(
                id: callId,
                toolName: name,
                providerMetadata: ["openai": ["item": item]]
            )]

        case "reasoning":
            // Deltas may or may not follow, so the block opens lazily on the
            // first one rather than here.
            pendingReasoning[itemKey] = item
            return []

        case "message":
            return []

        case let type where !Self.ownItemTypes.contains(type):
            // Provider-executed tool activity: web search, code interpreter,
            // shell, MCP, apply patch, image generation. The item type is the
            // tool name — there is no separate name field.
            //
            // Surfacing these as tool calls with `providerExecuted` set is not
            // cosmetic: a shell or apply-patch call mistaken for a client one
            // would be executed a second time, with real side effects.
            let callId = item["call_id"]?.stringValue ?? item["id"]?.stringValue ?? itemKey
            pendingServerTools[itemKey] = PendingToolCall(
                callId: callId, name: type, namespace: nil, arguments: "", itemType: type
            )
            return [.toolInputStart(
                id: callId,
                toolName: type,
                providerExecuted: true,
                dynamic: type.hasPrefix("mcp_")
            )]

        default:
            return []
        }
    }

    /// Routes a `response.<tool>.delta` event to the item it belongs to.
    private mutating func serverToolDelta(_ chunk: JSONValue) -> [StreamPart] {
        guard let key = itemKey(chunk), let delta = chunk["delta"]?.stringValue else {
            return [Self.malformed(
                chunk["type"]?.stringValue ?? "provider tool delta",
                "missing item_id/output_index or string delta", chunk
            )]
        }
        guard !delta.isEmpty else { return [] }
        guard let pending = pendingServerTools[key] else { return [] }

        pendingServerTools[key]?.arguments = pending.arguments + delta
        return [.toolInputDelta(id: pending.callId, delta: delta)]
    }

    /// Closes a provider-executed item, emitting both the call and its result.
    ///
    /// Unlike a client tool — where the call is all the provider sends and the
    /// result comes back from the caller — a server tool reports both halves,
    /// so both are emitted here.
    private mutating func finishServerTool(
        _ item: JSONValue,
        itemId: String,
        pending: PendingToolCall
    ) -> [StreamPart] {
        // Input may have streamed as deltas or arrived whole on the item.
        // Whichever it is, the assembled value must be valid JSON.
        var input = pending.arguments
        if input.isEmpty {
            input = Self.inputPayload(of: item).flatMap { try? $0.encodedString() } ?? "{}"
        } else if (try? JSONValue.decode(from: input)) == nil {
            // Several tools stream a bare fragment — a shell command, a patch
            // diff, a block of code — rather than JSON. Wrapping keeps the
            // contract that assembled input parses.
            input = (try? JSONValue.object(["input": .string(input)]).encodedString()) ?? "{}"
        }

        var parts: [StreamPart] = [
            .toolInputEnd(id: pending.callId),
            .toolCall(ToolCall(
                toolCallId: pending.callId,
                toolName: pending.name,
                namespace: pending.namespace,
                input: input,
                providerExecuted: true,
                dynamic: pending.name.hasPrefix("mcp_")
            )),
        ]

        parts.append(.toolResult(ToolResult(
            toolCallId: pending.callId,
            toolName: pending.name,
            // The whole item is the result: its payload shape is specific to
            // the tool, and flattening sixteen unrelated tools into a common
            // shape would discard most of what each reports.
            result: item,
            isError: item["status"]?.stringValue == "failed" || item["error"] != nil,
            dynamic: pending.name.hasPrefix("mcp_"),
            providerMetadata: ["openai": ["itemType": .string(pending.name)]]
        )))

        // Search results double as citable sources.
        parts.append(contentsOf: Self.sources(of: item, callId: pending.callId))

        return parts
    }

    /// The field carrying a server tool's input, which each tool names
    /// differently.
    private static func inputPayload(of item: JSONValue) -> JSONValue? {
        for key in ["arguments", "action", "operation", "code", "query"] {
            if let value = item[key], !value.isNull { return value }
        }
        return nil
    }

    private static func sources(of item: JSONValue, callId: String) -> [StreamPart] {
        var parts: [StreamPart] = []
        for (index, result) in (item["results"]?.arrayValue ?? []).enumerated() {
            guard let url = result["url"]?.stringValue else { continue }
            parts.append(.source(Source(
                id: "\(callId)-\(index)",
                kind: .url(url),
                title: result["title"]?.stringValue
            )))
        }
        return parts
    }

    private mutating func outputItemDone(_ chunk: JSONValue) -> [StreamPart] {
        guard let item = chunk["item"]?.objectValue.map(JSONValue.object) else {
            return [Self.malformed("response.output_item.done", "missing object field `item`", chunk)]
        }
        guard item["type"]?.stringValue != nil else {
            return [Self.malformed(
                "response.output_item.done", "item is missing string field `type`", chunk
            )]
        }
        guard let itemKey = itemKey(chunk, item: item) else {
            return [Self.malformed(
                "response.output_item.done", "item needs `id` or event needs `output_index`", chunk
            )]
        }

        var parts: [StreamPart] = []

        if item["type"]?.stringValue == "reasoning" {
            let metadata: ProviderMetadata = ["openai": ["item": item]]
            if reasoningOpen.remove(itemKey) != nil {
                parts.append(.reasoningEnd(id: itemKey, providerMetadata: metadata))
            } else if item["encrypted_content"] != nil || !(item["summary"]?.arrayValue ?? []).isEmpty {
                // Encrypted reasoning can have no readable summary deltas. It
                // still needs a block so aggregate history retains it.
                parts.append(.reasoningStart(id: itemKey, providerMetadata: metadata))
                parts.append(.reasoningEnd(id: itemKey, providerMetadata: metadata))
            }
            pendingReasoning.removeValue(forKey: itemKey)
        }

        if let pending = pendingServerTools.removeValue(forKey: itemKey) {
            parts.append(contentsOf: finishServerTool(item, itemId: itemKey, pending: pending))
        }

        if var pending = pendingToolCalls.removeValue(forKey: itemKey) {
            emittedClientToolCall = true
            toolCallOrder.removeAll { $0 == itemKey }
            let inputField = pending.itemType == "custom_tool_call" ? "input" : "arguments"
            if let completedArguments = item[inputField]?.stringValue {
                pending.arguments = completedArguments
            }
            parts.append(.toolInputEnd(id: pending.callId))
            parts.append(.toolCall(ToolCall(
                toolCallId: pending.callId,
                toolName: pending.name,
                namespace: pending.namespace,
                input: pending.itemType == "custom_tool_call"
                    ? pending.arguments
                    : (pending.arguments.isEmpty ? "{}" : pending.arguments),
                providerMetadata: ["openai": ["item": item]]
            )))
        }

        return parts
    }

    private mutating func contentPartAdded(_ chunk: JSONValue) -> [StreamPart] {
        guard let partType = chunk["part"]?["type"]?.stringValue else {
            return [Self.malformed(
                "response.content_part.added", "missing part.type", chunk
            )]
        }
        guard partType == "output_text" else { return [] }
        guard let id = textId(chunk) else {
            return [Self.malformed(
                "response.content_part.added", "missing item_id or content_index", chunk
            )]
        }

        guard !textOpen.contains(id) else { return [] }
        textOpen.insert(id)
        return [.textStart(id: id)]
    }

    private mutating func reasoningDelta(_ chunk: JSONValue) -> [StreamPart] {
        guard let itemId = itemKey(chunk), let delta = chunk["delta"]?.stringValue else {
            return [Self.malformed(
                "response.reasoning_summary_text.delta", "missing item_id/output_index or string delta", chunk
            )]
        }
        guard !delta.isEmpty else { return [] }

        var parts: [StreamPart] = []
        if !reasoningOpen.contains(itemId) {
            reasoningOpen.insert(itemId)
            let metadata = pendingReasoning[itemId].map { ["openai": ["item": $0]] }
            parts.append(.reasoningStart(id: itemId, providerMetadata: metadata))
        }
        parts.append(.reasoningDelta(id: itemId, delta: delta))
        return parts
    }

    private mutating func functionArgumentsDelta(_ chunk: JSONValue) -> [StreamPart] {
        guard let itemId = itemKey(chunk), let delta = chunk["delta"]?.stringValue else {
            return [Self.malformed(
                "response.function_call_arguments.delta", "missing item_id/output_index or string delta", chunk
            )]
        }
        guard !delta.isEmpty else { return [] }
        guard let pending = pendingToolCalls[itemId] else { return [] }

        pendingToolCalls[itemId]?.arguments = pending.arguments + delta
        return [.toolInputDelta(id: pending.callId, delta: delta)]
    }

    private mutating func clientToolInputDelta(_ chunk: JSONValue) -> [StreamPart] {
        guard let itemId = itemKey(chunk), let delta = chunk["delta"]?.stringValue else {
            return [Self.malformed(
                "response.custom_tool_call_input.delta",
                "missing item_id/output_index or string delta", chunk
            )]
        }
        guard !delta.isEmpty else { return [] }
        guard let pending = pendingToolCalls[itemId] else { return [] }

        pendingToolCalls[itemId]?.arguments = pending.arguments + delta
        return [.toolInputDelta(id: pending.callId, delta: delta)]
    }

    private mutating func functionArgumentsDone(_ chunk: JSONValue) -> [StreamPart] {
        guard let itemId = itemKey(chunk), let arguments = chunk["arguments"]?.stringValue else {
            return [Self.malformed(
                "response.function_call_arguments.done", "missing item_id/output_index or string arguments", chunk
            )]
        }
        if pendingToolCalls[itemId]?.arguments.isEmpty == true {
            pendingToolCalls[itemId]?.arguments = arguments
        }
        return []
    }

    private mutating func customToolInputDone(_ chunk: JSONValue) -> [StreamPart] {
        guard let itemId = itemKey(chunk), let input = chunk["input"]?.stringValue else {
            return [Self.malformed(
                "response.custom_tool_call_input.done",
                "missing item_id/output_index or string input", chunk
            )]
        }
        pendingToolCalls[itemId]?.arguments = input
        return []
    }

    private mutating func responseTerminal(
        _ chunk: JSONValue,
        emitFailureError: Bool = true
    ) -> [StreamPart] {
        guard let response = chunk["response"]?.objectValue.map(JSONValue.object) else {
            return [Self.malformed(
                chunk["type"]?.stringValue ?? "response terminal",
                "missing object field `response`", chunk
            )]
        }
        terminalResponse = response

        guard let status = response["status"]?.stringValue else {
            return [Self.malformed(
                chunk["type"]?.stringValue ?? "response terminal",
                "response is missing string field `status`", chunk
            )]
        }

        if let responseUsage = response["usage"], !responseUsage.isNull {
            usage = responseUsage
        }

        finishReason = Self.mapStatus(
            status,
            incompleteReason: response["incomplete_details"]?["reason"]?.stringValue,
            hasToolCalls: emittedClientToolCall || !toolCallOrder.isEmpty
        )

        guard emitFailureError, status == "failed" else { return [] }
        guard let error = response["error"], !error.isNull else {
            return [Self.malformed("response.failed", "missing response.error details", chunk)]
        }
        return [.error(StreamError(
            type: error["code"]?.stringValue ?? error["type"]?.stringValue,
            message: error["message"]?.stringValue ?? "Unknown provider error",
            raw: chunk
        ))]
    }

    private static func validationError(for type: String, chunk: JSONValue) -> String? {
        guard let schema = eventSchemas[type] else { return nil }

        for field in schema {
            guard let value = value(in: chunk, at: field.path) else {
                if field.required {
                    return "missing \(field.kind.rawValue) field `\(field.name)`"
                }
                continue
            }
            guard field.kind.accepts(value) else {
                return "field `\(field.name)` must be \(field.kind.rawValue)"
            }
        }

        if type == "error" {
            return validateProviderError(chunk)
        }
        if type == "response.output_item.added" || type == "response.output_item.done" {
            return validateOutputItem(chunk["item"], prefix: "item", context: .event)
        }
        if type == "response.content_part.added" || type == "response.content_part.done" {
            return validateContentPart(chunk["part"], prefix: "part")
        }
        if type == "response.completed" || type == "response.incomplete" || type == "response.failed" {
            let expectedStatus = String(type.dropFirst("response.".count))
            let actualStatus = chunk["response"]?["status"]?.stringValue
            guard actualStatus == expectedStatus else {
                return "event type requires `response.status` to equal `\(expectedStatus)`"
            }
            for (index, item) in (chunk["response"]?["output"]?.arrayValue ?? []).enumerated() {
                if let error = validateOutputItem(
                    item,
                    prefix: "response.output[\(index)]",
                    context: .terminal
                ) {
                    return error
                }
            }
            if type == "response.failed" {
                guard chunk["response"]?["error"]?.objectValue != nil else {
                    return "missing object field `response.error`"
                }
                guard chunk["response"]?["error"]?["message"]?.stringValue != nil else {
                    return "missing string field `response.error.message`"
                }
            }
        }

        return nil
    }

    private static func codexTerminalAliasValidationError(
        _ chunk: JSONValue,
        allowedStatuses: Set<String>
    ) -> String? {
        guard chunk["sequence_number"]?.intValue != nil else {
            return chunk["sequence_number"] == nil
                ? "missing integer field `sequence_number`"
                : "field `sequence_number` must be integer"
        }
        guard let response = chunk["response"], response.objectValue != nil else {
            return chunk["response"] == nil
                ? "missing object field `response`"
                : "field `response` must be object"
        }
        guard response["id"]?.stringValue != nil else {
            return response["id"] == nil
                ? "missing string field `response.id`"
                : "field `response.id` must be string"
        }
        guard let status = response["status"]?.stringValue else {
            return response["status"] == nil
                ? "missing string field `response.status`"
                : "field `response.status` must be string"
        }
        guard allowedStatuses.contains(status) else {
            return "field `response.status` has unsupported Codex value `\(status)`"
        }
        guard let output = response["output"]?.arrayValue else {
            return response["output"] == nil
                ? "missing array field `response.output`"
                : "field `response.output` must be array"
        }
        for (index, item) in output.enumerated() {
            if let error = validateOutputItem(
                item,
                prefix: "response.output[\(index)]",
                context: .terminal
            ) {
                return error
            }
        }
        return nil
    }

    /// Output item variants currently documented by OpenAI or present in the
    /// recorded fixture corpus. Unknown future variants remain valid and are
    /// retained; every variant in this catalog receives subtype validation.
    static let knownOutputItemTypes: Set<String> = [
        "message", "file_search_call", "function_call", "function_call_output",
        "web_search_call", "computer_call", "computer_call_output", "reasoning",
        "program", "program_output", "tool_search_call", "tool_search_output",
        "additional_tools", "compaction", "image_generation_call", "code_interpreter_call",
        "local_shell_call", "local_shell_call_output", "shell_call", "shell_call_output",
        "apply_patch_call", "apply_patch_call_output", "mcp_call", "mcp_list_tools",
        "mcp_approval_request", "mcp_approval_response", "custom_tool_call",
        "custom_tool_call_output",
    ]

    private enum OutputItemContext: Equatable {
        case event
        case terminal
    }

    private static func validateOutputItem(
        _ item: JSONValue?,
        prefix: String,
        context: OutputItemContext
    ) -> String? {
        guard let item, item.objectValue != nil else {
            return "missing object field `\(prefix)`"
        }
        guard let type = item["type"]?.stringValue else {
            return "missing string field `\(prefix).type`"
        }

        for (field, kind) in [
            ("id", JSONKind.string),
            ("status", JSONKind.string),
            ("phase", JSONKind.string),
        ] {
            if let error = optional(item, field, kind, prefix: prefix) { return error }
        }

        // A new discriminator is a valid forward-compatible item. The known
        // catalog is exhaustive by construction and independently asserted by
        // tests, so a known subtype can never silently take this path.
        guard knownOutputItemTypes.contains(type) else { return nil }

        switch type {
        case "message":
            if let error = require(item, "role", .string, prefix: prefix) { return error }
            if let error = require(item, "status", .string, prefix: prefix) { return error }
            if let error = require(item, "content", .array, prefix: prefix) { return error }
            for (index, part) in (item["content"]?.arrayValue ?? []).enumerated() {
                if let error = validateContentPart(part, prefix: "\(prefix).content[\(index)]") {
                    return error
                }
            }

        case "function_call":
            for field in ["call_id", "name", "arguments"] {
                if let error = require(item, field, .string, prefix: prefix) { return error }
            }
            if let error = optional(item, "namespace", .string, prefix: prefix) { return error }

        case "custom_tool_call":
            for field in ["call_id", "name", "input"] {
                if let error = require(item, field, .string, prefix: prefix) { return error }
            }
            if let error = optional(item, "namespace", .string, prefix: prefix) { return error }

        case "function_call_output":
            // The provider's output-item schema permits call_id to be absent
            // on retained output items even though callers normally include it
            // when sending a tool result as input.
            if let error = optional(item, "call_id", .string, prefix: prefix) { return error }
            if let error = validateToolOutput(item["output"], prefix: "\(prefix).output") { return error }

        case "custom_tool_call_output":
            if let error = require(item, "call_id", .string, prefix: prefix) { return error }
            if let error = validateToolOutput(item["output"], prefix: "\(prefix).output") { return error }

        case "reasoning":
            if let error = require(item, "summary", .array, prefix: prefix) { return error }
            if let error = optional(item, "encrypted_content", .string, prefix: prefix) { return error }
            if let error = optional(item, "content", .array, prefix: prefix) { return error }
            for (index, summary) in (item["summary"]?.arrayValue ?? []).enumerated() {
                if let error = validateSummaryPart(summary, prefix: "\(prefix).summary[\(index)]") {
                    return error
                }
            }
            for (index, part) in (item["content"]?.arrayValue ?? []).enumerated() {
                if let error = validateContentPart(part, prefix: "\(prefix).content[\(index)]") {
                    return error
                }
            }

        case "compaction":
            if context == .terminal {
                if let error = require(item, "encrypted_content", .string, prefix: prefix) { return error }
            } else if let error = optional(item, "encrypted_content", .string, prefix: prefix) {
                return error
            }

        case "file_search_call":
            if let error = require(item, "queries", .array, prefix: prefix) { return error }
            if let error = require(item, "status", .string, prefix: prefix) { return error }
            if let error = optionalNullable(item, "results", .array, prefix: prefix) { return error }

        case "web_search_call":
            if let error = require(item, "status", .string, prefix: prefix) { return error }
            if context == .terminal {
                if let error = require(item, "action", .object, prefix: prefix) { return error }
            } else if let error = optional(item, "action", .object, prefix: prefix) {
                return error
            }

        case "computer_call":
            if let error = require(item, "call_id", .string, prefix: prefix) { return error }
            if let error = require(item, "pending_safety_checks", .array, prefix: prefix) { return error }
            if let error = require(item, "status", .string, prefix: prefix) { return error }
            if let error = optional(item, "action", .object, prefix: prefix) { return error }
            if let error = optional(item, "actions", .array, prefix: prefix) { return error }

        case "computer_call_output":
            if let error = require(item, "call_id", .string, prefix: prefix) { return error }
            if let error = require(item, "output", .object, prefix: prefix) { return error }
            if let error = require(item, "status", .string, prefix: prefix) { return error }

        case "program":
            if let error = require(item, "call_id", .string, prefix: prefix) { return error }
            if let error = require(item, "code", .string, prefix: prefix) { return error }
            if context == .terminal {
                if let error = require(item, "fingerprint", .string, prefix: prefix) { return error }
            } else if let error = optional(item, "fingerprint", .string, prefix: prefix) {
                return error
            }

        case "program_output":
            if let error = require(item, "call_id", .string, prefix: prefix) { return error }
            if let error = require(item, "status", .string, prefix: prefix) { return error }
            if context == .terminal {
                if let error = require(item, "result", .string, prefix: prefix) { return error }
            } else if let error = optional(item, "result", .string, prefix: prefix) {
                return error
            }

        case "tool_search_call":
            if let error = require(item, "arguments", .object, prefix: prefix) { return error }
            if let error = requireNullableString(item, "call_id", prefix: prefix) { return error }
            if let error = require(item, "execution", .string, prefix: prefix) { return error }
            if let error = require(item, "status", .string, prefix: prefix) { return error }

        case "tool_search_output":
            if let error = requireNullableString(item, "call_id", prefix: prefix) { return error }
            if let error = require(item, "execution", .string, prefix: prefix) { return error }
            if let error = require(item, "status", .string, prefix: prefix) { return error }
            if let error = require(item, "tools", .array, prefix: prefix) { return error }

        case "additional_tools":
            if let error = require(item, "role", .string, prefix: prefix) { return error }
            if let error = require(item, "tools", .array, prefix: prefix) { return error }

        case "image_generation_call":
            if let error = require(item, "status", .string, prefix: prefix) { return error }
            if let error = optional(item, "result", .string, prefix: prefix) { return error }

        case "code_interpreter_call":
            if let error = require(item, "container_id", .string, prefix: prefix) { return error }
            if let error = require(item, "status", .string, prefix: prefix) { return error }
            if let error = optionalNullable(item, "code", .string, prefix: prefix) { return error }
            if let error = optionalNullable(item, "outputs", .array, prefix: prefix) { return error }

        case "local_shell_call":
            if let error = require(item, "action", .object, prefix: prefix) { return error }
            if let error = require(item, "call_id", .string, prefix: prefix) { return error }
            if let error = require(item, "status", .string, prefix: prefix) { return error }

        case "local_shell_call_output":
            if let error = require(item, "output", .string, prefix: prefix) { return error }

        case "shell_call":
            if let error = require(item, "action", .object, prefix: prefix) { return error }
            if let error = require(item, "call_id", .string, prefix: prefix) { return error }
            if let error = require(item, "status", .string, prefix: prefix) { return error }
            if let error = optionalNullable(item, "environment", .object, prefix: prefix) { return error }

        case "shell_call_output":
            if let error = require(item, "call_id", .string, prefix: prefix) { return error }
            if let error = require(item, "output", .array, prefix: prefix) { return error }
            if let error = require(item, "status", .string, prefix: prefix) { return error }

        case "apply_patch_call":
            if let error = require(item, "call_id", .string, prefix: prefix) { return error }
            if let error = require(item, "operation", .object, prefix: prefix) { return error }
            if let error = require(item, "status", .string, prefix: prefix) { return error }

        case "apply_patch_call_output":
            if let error = require(item, "call_id", .string, prefix: prefix) { return error }
            if let error = require(item, "status", .string, prefix: prefix) { return error }
            if let error = optional(item, "output", .string, prefix: prefix) { return error }

        case "mcp_call":
            for field in ["arguments", "name", "server_label"] {
                if let error = require(item, field, .string, prefix: prefix) { return error }
            }
            if let error = optionalNullable(item, "output", .string, prefix: prefix) { return error }
            if let error = optionalMCPError(item, prefix: prefix) { return error }

        case "mcp_list_tools":
            if let error = require(item, "server_label", .string, prefix: prefix) { return error }
            if let error = require(item, "tools", .array, prefix: prefix) { return error }
            if let error = optionalNullable(item, "error", .string, prefix: prefix) { return error }

        case "mcp_approval_request":
            for field in ["arguments", "name", "server_label"] {
                if let error = require(item, field, .string, prefix: prefix) { return error }
            }

        case "mcp_approval_response":
            if let error = require(item, "approval_request_id", .string, prefix: prefix) { return error }
            if let error = require(item, "approve", .boolean, prefix: prefix) { return error }

        default:
            // Exhaustive over `knownOutputItemTypes`; retained for source
            // compatibility when the catalog is extended before this switch.
            return "known output item type `\(type)` has no validator"
        }

        return nil
    }

    private static func validateContentPart(_ part: JSONValue?, prefix: String) -> String? {
        guard let part, part.objectValue != nil else { return "missing object field `\(prefix)`" }
        guard let type = part["type"]?.stringValue else {
            return "missing string field `\(prefix).type`"
        }

        switch type {
        case "output_text":
            if let error = require(part, "text", .string, prefix: prefix) { return error }
            if let error = optional(part, "annotations", .array, prefix: prefix) { return error }
            if let error = optional(part, "logprobs", .array, prefix: prefix) { return error }
        case "refusal":
            if let error = require(part, "refusal", .string, prefix: prefix) { return error }
        case "reasoning_text":
            if let error = require(part, "text", .string, prefix: prefix) { return error }
        default:
            break
        }
        return nil
    }

    private static func validateSummaryPart(_ part: JSONValue, prefix: String) -> String? {
        guard part.objectValue != nil else { return "field `\(prefix)` must be object" }
        guard let type = part["type"]?.stringValue else {
            return "missing string field `\(prefix).type`"
        }
        guard type == "summary_text" else { return nil }
        return require(part, "text", .string, prefix: prefix)
    }

    private static func validateToolOutput(_ output: JSONValue?, prefix: String) -> String? {
        guard let output else { return "missing string or array field `\(prefix)`" }
        if output.stringValue != nil { return nil }
        guard let parts = output.arrayValue else {
            return "field `\(prefix)` must be string or array"
        }
        for (index, part) in parts.enumerated() {
            let partPrefix = "\(prefix)[\(index)]"
            guard part.objectValue != nil else { return "field `\(partPrefix)` must be object" }
            guard let type = part["type"]?.stringValue else {
                return "missing string field `\(partPrefix).type`"
            }
            switch type {
            case "input_text":
                if let error = require(part, "text", .string, prefix: partPrefix) { return error }
            case "input_image":
                if let error = optionalNullable(part, "detail", .string, prefix: partPrefix) { return error }
                if let error = optionalNullable(part, "image_url", .string, prefix: partPrefix) { return error }
                if let error = optionalNullable(part, "file_id", .string, prefix: partPrefix) { return error }
                guard part["image_url"]?.stringValue != nil || part["file_id"]?.stringValue != nil else {
                    return "`\(partPrefix)` needs string image_url or file_id"
                }
            case "input_file":
                if let error = optional(part, "detail", .string, prefix: partPrefix) { return error }
                for field in ["file_data", "file_id", "file_url", "filename"] {
                    if let error = optionalNullable(part, field, .string, prefix: partPrefix) { return error }
                }
                guard ["file_data", "file_id", "file_url"].contains(where: {
                    part[$0]?.stringValue != nil
                }) else {
                    return "`\(partPrefix)` needs string file_data, file_id, or file_url"
                }
            default:
                // Unknown future content discriminators remain valid, but the
                // envelope is still recursively shape-checked above.
                break
            }
        }
        return nil
    }

    private static func require(
        _ object: JSONValue,
        _ field: String,
        _ kind: JSONKind,
        prefix: String
    ) -> String? {
        guard let value = object[field] else {
            return "missing \(kind.rawValue) field `\(prefix).\(field)`"
        }
        guard kind.accepts(value) else {
            return "field `\(prefix).\(field)` must be \(kind.rawValue)"
        }
        return nil
    }

    private static func optional(
        _ object: JSONValue,
        _ field: String,
        _ kind: JSONKind,
        prefix: String
    ) -> String? {
        guard let value = object[field] else { return nil }
        guard kind.accepts(value) else {
            return "field `\(prefix).\(field)` must be \(kind.rawValue)"
        }
        return nil
    }

    private static func optionalNullable(
        _ object: JSONValue,
        _ field: String,
        _ kind: JSONKind,
        prefix: String
    ) -> String? {
        guard let value = object[field], !value.isNull else { return nil }
        guard kind.accepts(value) else {
            return "field `\(prefix).\(field)` must be \(kind.rawValue) or null"
        }
        return nil
    }

    private static func requireNullableString(
        _ object: JSONValue,
        _ field: String,
        prefix: String
    ) -> String? {
        guard let value = object[field] else {
            return "missing string or null field `\(prefix).\(field)`"
        }
        guard value.isNull || value.stringValue != nil else {
            return "field `\(prefix).\(field)` must be string or null"
        }
        return nil
    }

    private static func optionalMCPError(_ object: JSONValue, prefix: String) -> String? {
        guard let value = object["error"], !value.isNull else { return nil }
        guard value.stringValue != nil || value.objectValue != nil else {
            return "field `\(prefix).error` must be string, object, or null"
        }
        return nil
    }

    private static func validateProviderError(_ chunk: JSONValue) -> String? {
        if let error = chunk["error"] {
            guard error.objectValue != nil else { return "field `error` must be object" }
            guard error["message"]?.stringValue != nil else {
                return "missing string field `error.message`"
            }
            if let value = error["code"], !value.isNull, value.stringValue == nil {
                return "field `error.code` must be string or null"
            }
            if let value = error["type"], !value.isNull, value.stringValue == nil {
                return "field `error.type` must be string or null"
            }
            return nil
        }

        if let code = chunk["code"], !code.isNull, code.stringValue == nil {
            return "field `code` must be string or null"
        }
        guard chunk["message"]?.stringValue != nil else {
            return "missing string field `message`"
        }
        if let param = chunk["param"], !param.isNull, param.stringValue == nil {
            return "field `param` must be string or null"
        }
        return nil
    }

    private static func value(in root: JSONValue, at path: [String]) -> JSONValue? {
        path.reduce(Optional(root)) { value, component in value?[component] }
    }

    private static func malformed(_ event: String, _ detail: String, _ chunk: JSONValue) -> StreamPart {
        .error(StreamError(
            type: "malformed_event",
            message: "Malformed known OpenAI Responses event `\(event)`: \(detail)",
            raw: chunk
        ))
    }

    private static func providerError(_ chunk: JSONValue) -> [StreamPart] {
        let nestedError = chunk["error"]?.objectValue.map(JSONValue.object)
        let error = nestedError ?? chunk
        guard let message = error["message"]?.stringValue else {
            return [malformed("error", "missing provider error message", chunk)]
        }
        return [.error(StreamError(
            // Top-level `type` is only the event discriminator. The captured
            // nested shape instead uses `error.type` as an error category.
            type: error["code"]?.stringValue
                ?? (nestedError == nil ? nil : error["type"]?.stringValue),
            message: message,
            raw: chunk
        ))]
    }

    // MARK: - Helpers

    /// Text blocks are addressed by item *and* content index, since one item
    /// can carry several content parts.
    private func textId(_ chunk: JSONValue) -> String? {
        guard let itemId = chunk["item_id"]?.stringValue,
              let contentIndex = chunk["content_index"]?.intValue else { return nil }
        return "\(itemId)#\(contentIndex)"
    }

    private func itemKey(_ chunk: JSONValue, item: JSONValue? = nil) -> String? {
        if let id = item?["id"]?.stringValue ?? chunk["item_id"]?.stringValue { return id }
        if let index = chunk["output_index"]?.intValue { return "output-index:\(index)" }
        return nil
    }

    private var finishMetadata: ProviderMetadata? {
        var metadata: [String: JSONValue] = [:]
        if let usage { metadata["usage"] = usage }
        if let terminalResponse { metadata["response"] = terminalResponse }
        return metadata.isEmpty ? nil : ["openai": metadata]
    }

    private mutating func closeOpenBlocks() -> [StreamPart] {
        var parts: [StreamPart] = []

        for id in reasoningOpen.sorted() { parts.append(.reasoningEnd(id: id)) }
        reasoningOpen.removeAll()

        for id in textOpen.sorted() { parts.append(.textEnd(id: id)) }
        textOpen.removeAll()

        return parts
    }

    private mutating func flushPendingToolCalls() -> [StreamPart] {
        guard !pendingToolCalls.isEmpty else { return [] }

        var parts: [StreamPart] = []
        for itemId in toolCallOrder {
            guard let pending = pendingToolCalls[itemId] else { continue }
            parts.append(.toolInputEnd(id: pending.callId))
            parts.append(.toolCall(ToolCall(
                toolCallId: pending.callId,
                toolName: pending.name,
                namespace: pending.namespace,
                input: pending.itemType == "custom_tool_call"
                    ? pending.arguments
                    : (pending.arguments.isEmpty ? "{}" : pending.arguments)
            )))
        }

        pendingToolCalls.removeAll()
        toolCallOrder.removeAll()
        return parts
    }

    // MARK: - Mapping tables

    static func mapStatus(
        _ status: String?,
        incompleteReason: String? = nil,
        hasToolCalls: Bool = false
    ) -> FinishReason {
        let unified: FinishReason.Unified
        switch status {
        case "completed":
            // The Responses API reports success without saying whether tools
            // are pending, so the caller's only signal is the call itself.
            unified = hasToolCalls ? .toolCalls : .stop
        case "incomplete":
            unified = incompleteReason == "max_output_tokens" ? .length : .contentFilter
        case "failed", "cancelled":
            unified = .error
        case "queued", "in_progress":
            unified = .stop
        default:
            unified = .other
        }
        return FinishReason(unified: unified, raw: status)
    }

    /// - Important: like Chat Completions and unlike Anthropic, `input_tokens`
    ///   here **includes** cached tokens, and `output_tokens` **includes**
    ///   reasoning tokens.
    static func convertUsage(_ usage: JSONValue) -> Usage {
        let inputTokens = usage["input_tokens"]?.intValue
        let outputTokens = usage["output_tokens"]?.intValue
        let cachedTokens = usage["input_tokens_details"]?["cached_tokens"]?.intValue ?? 0
        let reasoningTokens = usage["output_tokens_details"]?["reasoning_tokens"]?.intValue

        return Usage(
            inputTokens: .init(
                total: inputTokens,
                noCache: inputTokens.map { $0 - cachedTokens },
                cacheRead: cachedTokens,
                cacheWrite: 0
            ),
            outputTokens: .init(
                total: outputTokens,
                text: {
                    guard let outputTokens, let reasoningTokens else { return nil }
                    return outputTokens - reasoningTokens
                }(),
                reasoning: reasoningTokens
            ),
            raw: usage
        )
    }
}
