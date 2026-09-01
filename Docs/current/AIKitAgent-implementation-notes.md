Yes—this maps cleanly:

```text
AIKit / AIFacade       ≈ pi-ai
AIKitAgent / Agent     ≈ pi-agent-core
```

`AIFacade` should remain the immutable provider/model transport layer. A new stateful `Agent` should orchestrate conversations and tools on top of `AIFacade.stream(_:)`.

## Recommended structure

Add a separate `AIKitAgent` target depending on `AIKit`:

```text
Sources/
  AIKit/               existing provider-neutral model layer
  AIKitAgent/
    Agent.swift        stateful actor facade
    AgentLoop.swift    model/tool loop
    AgentTypes.swift   state, events, run results
    AgentTool.swift    executable tool abstraction
```

Keeping it separate mirrors pi’s boundary and prevents session/tool orchestration from leaking into the wire layer.

## Public API shape

```swift
public actor Agent {
    public init(
        facade: AIFacade,
        initialState: AgentState,
        options: AgentOptions = .init()
    )

    public var state: AgentState { get }

    public func prompt(_ text: String) async throws -> AgentRunResult
    public func prompt(_ messages: [Message]) async throws -> AgentRunResult
    public func resume() async throws -> AgentRunResult

    public func steer(_ message: Message)
    public func followUp(_ message: Message)

    public func abort()
    public func reset() throws

    @discardableResult
    public func subscribe(
        _ handler: @escaping @Sendable (AgentEvent) async -> Void
    ) -> AgentSubscriptionID
}
```

Unlike pi’s mutable `agent.state`, Swift should expose an immutable snapshot with actor-isolated mutation methods.

```swift
public struct AgentState: Sendable {
    public var systemPrompt: String
    public var destination: ModelDestination
    public var thinking: Thinking?
    public var tools: [AgentTool]
    public var messages: [Message]

    public private(set) var isRunning: Bool
    public private(set) var streamingResponse: AIResponse?
    public private(set) var pendingToolCallIDs: Set<String>
    public private(set) var errorMessage: String?
}
```

## Executable tools

Keep the existing `ToolDefinition` as the provider-facing schema and wrap it with runtime behaviour:

```swift
public struct AgentTool: Sendable {
    public let definition: ToolDefinition
    public let label: String
    public let executionMode: ToolExecutionMode?

    public init<Arguments: Decodable & Sendable>(
        definition: ToolDefinition,
        label: String,
        arguments: Arguments.Type,
        execute: @escaping @Sendable (
            Arguments,
            AgentToolContext,
            AgentToolUpdate
        ) async throws -> AgentToolResult
    )
}
```

A typed initializer avoids handing every executor unvalidated `JSONValue`. Because AIKit has no JSON Schema validator, I would support:

1. mandatory valid-JSON decoding,
2. typed `Decodable` validation,
3. an optional custom schema-validation closure,

rather than implementing a partial JSON Schema validator internally.

## Events

Preserve pi-agent-core’s lifecycle, but retain AIKit’s normalized stream parts:

```swift
public enum AgentEvent: Sendable {
    case agentStart
    case agentEnd(AgentRunResult)

    case turnStart
    case turnEnd(response: AIResponse, toolResults: [Message])

    case messageStart(Message)
    case messageUpdate(message: Message, part: StreamPart)
    case messageEnd(Message)

    case toolExecutionStart(ToolCall)
    case toolExecutionUpdate(ToolCall, AgentToolResult)
    case toolExecutionEnd(ToolCall, AgentToolResult, isError: Bool)
}
```

Handlers should be awaited in registration order. State must be reduced before handlers run, ensuring a `messageEnd` handler and tool preflight see the finalized assistant message—an important pi-agent-core invariant.

## Loop semantics

Each run should:

1. Snapshot destination and request settings for the turn.
2. Build the prompt from system prompt plus transcript.
3. Call `AIFacade.stream(_:)`.
4. Accumulate `StreamPart` into `AIResponse`.
5. Append `response.assistantMessage`.
6. Execute `response.pendingToolCalls`.
7. Append tool-result messages in assistant source order.
8. Repeat until there are no local tool calls.
9. Drain steering before follow-ups.
10. Emit exactly one terminal `agentEnd`.

Important invariants from current pi-agent-core 0.84.4 worth retaining:

- Assistant `messageEnd` precedes tool preflight.
- Tool failures become `isError` tool results instead of crashing the loop.
- Parallel tools may finish out of order, but transcript results remain in source order.
- Tool calls from a `.length` response are not executed because their arguments may be truncated.
- Only terminate early when every tool result in the batch requests termination.
- Cancellation settles exactly once.

## Intentional Swift divergence

I would not copy pi’s ambiguous failure handling. Transport/runtime failures should:

- update `state.errorMessage`,
- emit a failed or aborted assistant message for observation,
- emit terminal lifecycle events,
- then throw from `prompt()`.

That gives callers normal Swift error handling while existing `ConversationTransformer` ensures failed history is not replayed.

## What should remain out of `Agent`

These belong in a later `AgentSession` or harness layer:

- persistence and branching,
- compaction,
- automatic retries,
- skills and prompt templates,
- built-in filesystem/shell tools,
- UI-specific observable state.

The existing `AIFacade`, `AIResponse.assistantMessage`, destination identity, and conversation transformation already provide the difficult model-layer foundation. The next logical step is an `AIKitAgent` target with a deterministic fake-stream-driven test suite before wiring real HTTP fixtures into the loop.
