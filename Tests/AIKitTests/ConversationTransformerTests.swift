import Foundation
import Testing

@testable import AIKit

@Suite("Conversation transformation")
struct ConversationTransformerTests {
    private let openAIResponses = ModelDestination(
        providerId: "openai",
        apiId: WireProtocol.openAIResponses.rawValue,
        modelId: "gpt-5"
    )

    private let opaqueReasoningItem: JSONValue = [
        "type": "reasoning",
        "id": "rs_opaque",
        "encrypted_content": "encrypted+/=bytes",
        "summary": [["type": "summary_text", "text": "readable"]],
        "status": "completed",
    ]

    private let opaqueCallItem: JSONValue = [
        "type": "function_call",
        "id": "fc_source+/=id",
        "call_id": "call_source",
        "name": "lookup",
        "namespace": "source_namespace",
        "arguments": #"{"value":1}"#,
        "status": "completed",
    ]

    @Test("an exact Responses identity preserves every opaque JSON value and identifier")
    func exactResponsesRoundTrip() throws {
        let call = ToolCall(
            toolCallId: "call_source",
            toolName: "lookup",
            namespace: "source_namespace",
            input: #"{"value":1}"#,
            providerMetadata: ["openai": ["item": opaqueCallItem]]
        )
        let prompt: Prompt = [
            Message(
                role: .assistant,
                content: [
                    .reasoning("readable", providerMetadata: [
                        "openai": ["item": opaqueReasoningItem]
                    ]),
                    .toolCall(call),
                    .text("answer"),
                ],
                providerOptions: [
                    "openai": [
                        "outputItems": .array([opaqueReasoningItem, opaqueCallItem]),
                        "futureState": ["bytes": "opaque+/="],
                    ]
                ],
                producer: openAIResponses
            ),
            Message.toolResult(
                toolCallId: "call_source",
                toolName: "lookup",
                content: [.text("done")]
            ),
        ]

        let transformed = transformer(for: openAIResponses).transform(prompt)

        #expect(transformed == prompt)
        let responsesClient = AIClient(
            provider: ProviderInfo(
                id: openAIResponses.providerId,
                api: "https://openai.example",
                adapter: openAIResponses.apiId,
                models: [ModelInfo(id: openAIResponses.modelId)]
            ),
            configuration: .init()
        )
        let responsesRequest = try responsesClient.prepare(CallOptions(
            model: openAIResponses.modelId,
            prompt: prompt
        ))
        #expect(responsesRequest.options.prompt == prompt)
        #expect(Array(
            (responsesClient.encode(responsesRequest).body["input"]?.arrayValue ?? []).prefix(2)
        ) == [opaqueReasoningItem, opaqueCallItem])

        let codex = ModelDestination(
            providerId: "openai-codex",
            apiId: WireProtocol.openAICodex.rawValue,
            modelId: "gpt-5-codex"
        )
        let codexPrompt = prompt.map { message -> Message in
            guard message.role == .assistant else { return message }
            var message = message
            message.producer = codex
            return message
        }
        #expect(transformer(for: codex).transform(codexPrompt) == codexPrompt)
        let codexClient = AIClient(
            provider: ProviderInfo(
                id: codex.providerId,
                api: "https://codex.example",
                adapter: codex.apiId,
                models: [ModelInfo(id: codex.modelId)]
            ),
            configuration: .init()
        )
        let codexRequest = try codexClient.prepare(CallOptions(
            model: codex.modelId,
            prompt: codexPrompt
        ))
        #expect(codexRequest.options.prompt == codexPrompt)
        #expect(Array(
            (codexClient.encode(codexRequest).body["input"]?.arrayValue ?? []).prefix(2)
        ) == [opaqueReasoningItem, opaqueCallItem])
    }

    @Test("exact Anthropic and Google identities preserve signed, redacted, and thought state")
    func exactAnthropicAndGoogleState() {
        let anthropic = ModelDestination(
            providerId: "anthropic",
            apiId: WireProtocol.anthropicMessages.rawValue,
            modelId: "claude"
        )
        let signed: ProviderMetadata = [
            "anthropic": [
                "signature": "signed+/=bytes",
                "wireBlock": [
                    "type": "thinking", "thinking": "visible", "signature": "signed+/=bytes",
                ],
            ]
        ]
        let redacted: ProviderMetadata = [
            "anthropic": [
                "blockType": "redacted_thinking",
                "redactedData": "redacted+/=bytes",
                "wireBlock": ["type": "redacted_thinking", "data": "redacted+/=bytes"],
            ]
        ]
        let anthropicPrompt: Prompt = [
            Message(
                role: .assistant,
                content: [
                    .reasoning("visible", providerMetadata: signed),
                    .reasoning("", providerMetadata: redacted),
                ],
                producer: anthropic
            )
        ]
        #expect(transformer(for: anthropic).transform(anthropicPrompt) == anthropicPrompt)

        let google = ModelDestination(
            providerId: "google",
            apiId: WireProtocol.googleGenerativeAI.rawValue,
            modelId: "gemini-3-pro"
        )
        let googlePrompt: Prompt = [
            Message(
                role: .assistant,
                content: [.toolCall(ToolCall(
                    toolCallId: "call-1",
                    toolName: "lookup",
                    input: "{}",
                    providerMetadata: ["google": ["thoughtSignature": "thought+/=bytes"]]
                ))],
                producer: google
            ),
            .toolResult(toolCallId: "call-1", toolName: "lookup", result: "done"),
        ]
        #expect(transformer(for: google).transform(googlePrompt) == googlePrompt)
    }

    @Test("provider, API, and model mismatches each downgrade and strip opaque state")
    func everyIdentityMismatchDimension() throws {
        let source = openAIResponses
        let call = ToolCall(
            toolCallId: "safe-call",
            toolName: "lookup",
            namespace: "source-only",
            input: "{}",
            providerMetadata: [
                "openai": ["item": opaqueCallItem],
                "google": ["thoughtSignature": "source-thought"],
            ]
        )
        let prompt: Prompt = [
            Message(
                role: .assistant,
                content: [
                    .text("before"),
                    .reasoning("readable reasoning", providerMetadata: [
                        "anthropic": ["signature": "source-signature"]
                    ]),
                    .reasoning("   ", providerMetadata: [
                        "anthropic": ["redactedData": "opaque"]
                    ]),
                    .toolCall(call),
                    .text("after"),
                ],
                providerOptions: ["openai": ["outputItems": [opaqueReasoningItem]]],
                producer: source,
                responseModelId: "server-fallback"
            ),
            Message(
                role: .tool,
                content: [.toolResult(ToolResult(
                    toolCallId: "safe-call",
                    toolName: "lookup",
                    result: "done",
                    providerMetadata: ["openai": ["output": ["private": true]]]
                ))]
            ),
        ]
        let destinations = [
            ModelDestination(
                providerId: source.providerId,
                apiId: WireProtocol.openAICompletions.rawValue,
                modelId: source.modelId
            ),
            ModelDestination(
                providerId: source.providerId,
                apiId: source.apiId,
                modelId: "gpt-5-mini"
            ),
            ModelDestination(
                providerId: "openai-codex",
                apiId: source.apiId,
                modelId: source.modelId
            ),
        ]

        for destination in destinations {
            let transformed = transformer(for: destination).transform(prompt)
            let assistant = try #require(transformed.first)
            #expect(assistant.providerOptions == nil)
            #expect(assistant.producer == source)
            #expect(assistant.responseModelId == "server-fallback")
            #expect(assistant.content.count == 4)
            #expect(text(in: assistant.content[0]) == "before")
            #expect(text(in: assistant.content[1]) == "readable reasoning")
            let transformedCall = try #require(toolCall(in: assistant))
            #expect(transformedCall.namespace == nil)
            #expect(transformedCall.providerMetadata?["openai"]?["item"] == nil)
            #expect(transformedCall.providerMetadata?["google"] == nil)
            #expect(text(in: assistant.content[3]) == "after")

            let result = try #require(toolResult(in: transformed))
            #expect(result.providerMetadata == nil)
            #expect(result.toolCallId == transformedCall.toolCallId)
        }
    }

    @Test("OpenAI and OpenAI Codex never share Responses replay state")
    func openAIAndCodexAreNotSameIdentity() throws {
        let codex = ModelDestination(
            providerId: "openai-codex",
            apiId: WireProtocol.openAICodex.rawValue,
            modelId: openAIResponses.modelId
        )
        let prompt: Prompt = [
            Message(
                role: .assistant,
                content: [.reasoning("summary", providerMetadata: [
                    "openai": ["item": opaqueReasoningItem]
                ])],
                providerOptions: ["openai": ["outputItems": [opaqueReasoningItem]]],
                producer: openAIResponses
            )
        ]

        let transformed = transformer(for: codex).transform(prompt)
        let assistant = try #require(transformed.first)
        #expect(assistant.providerOptions == nil)
        #expect(assistant.content == [.text("summary")])
    }

    @Test("destination ID rules preserve distinct calls and their linked results")
    func destinationSpecificToolCallIds() throws {
        let source = openAIResponses
        let calls = [
            ToolCall(
                toolCallId: "duplicate|unsafe+/=call",
                toolName: "one",
                namespace: "source",
                input: "{}",
                providerMetadata: ["openai": ["item": ["id": "fc/item-one"]]]
            ),
            ToolCall(
                toolCallId: "duplicate|unsafe+/=call",
                toolName: "two",
                namespace: "source",
                input: "{}",
                providerMetadata: ["openai": ["item": ["id": "fc/item-two"]]]
            ),
        ]
        let prompt: Prompt = [
            Message(role: .assistant, content: calls.map(ContentPart.toolCall), producer: source),
            resultMessage(call: calls[0], itemId: "fc/item-one"),
            resultMessage(call: calls[1], itemId: "fc/item-two"),
        ]

        let destinations = [
            ModelDestination(
                providerId: "anthropic",
                apiId: WireProtocol.anthropicMessages.rawValue,
                modelId: "claude"
            ),
            ModelDestination(
                providerId: "openai",
                apiId: WireProtocol.openAICompletions.rawValue,
                modelId: "gpt-4o"
            ),
            ModelDestination(
                providerId: "google",
                apiId: WireProtocol.googleGenerativeAI.rawValue,
                modelId: "gemini-3-pro"
            ),
        ]

        for destination in destinations {
            let transformed = transformer(for: destination).transform(prompt)
            let assistant = try #require(transformed.first)
            let mappedCalls = assistant.content.compactMap { part -> ToolCall? in
                if case .toolCall(let call) = part { call } else { nil }
            }
            let mappedResults = transformed.dropFirst().compactMap { message -> ToolResult? in
                message.content.compactMap {
                    if case .toolResult(let result) = $0 { result } else { nil }
                }.first
            }

            #expect(mappedCalls.count == 2)
            #expect(Set(mappedCalls.map(\.toolCallId)).count == 2)
            #expect(mappedResults.map(\.toolCallId) == mappedCalls.map(\.toolCallId))
            #expect(mappedCalls.allSatisfy { Self.isSafeID($0.toolCallId) })
            #expect(mappedCalls.allSatisfy { $0.providerMetadata == nil && $0.namespace == nil })
            #expect(mappedResults.allSatisfy { $0.providerMetadata == nil })
            let limit = destination.apiId == WireProtocol.openAICompletions.rawValue ? 40 : 64
            #expect(mappedCalls.allSatisfy { $0.toolCallId.count <= limit })
        }
    }

    @Test("Responses remaps foreign item identity without replaying its payload or id")
    func responsesSynthesizesSafeItemIdentity() throws {
        let source = ModelDestination(
            providerId: "github-copilot",
            apiId: WireProtocol.openAIResponses.rawValue,
            modelId: "foreign"
        )
        let call = ToolCall(
            toolCallId: "call|unsafe+/=",
            toolName: "lookup",
            namespace: "foreign",
            input: "{}",
            providerMetadata: ["openai": ["item": opaqueCallItem]]
        )
        let prompt: Prompt = [
            Message(role: .assistant, content: [.toolCall(call)], producer: source),
            .toolResult(toolCallId: call.toolCallId, toolName: call.toolName, result: "done"),
        ]

        let transformed = transformer(for: openAIResponses).transform(prompt)
        let assistant = try #require(transformed.first)
        let mapped = try #require(toolCall(in: assistant))
        let itemId = try #require(mapped.providerMetadata?["openai"]?["itemId"]?.stringValue)
        #expect(itemId.hasPrefix("fc_"))
        #expect(itemId != opaqueCallItem["id"]?.stringValue)
        #expect(mapped.providerMetadata?["openai"]?["item"] == nil)
        #expect(mapped.namespace == nil)
        #expect(Self.isSafeID(mapped.toolCallId))
        #expect(try #require(toolResult(in: transformed)).toolCallId == mapped.toolCallId)
        #expect(transformer(for: openAIResponses).transform(transformed) == transformed)
    }

    @Test("orphan closure handles existing, multiple, terminal, and provider-executed calls")
    func closesOnlyClientOrphans() throws {
        let destination = openAIResponses
        let first = ToolCall(toolCallId: "first", toolName: "first", input: "{}")
        let missing = ToolCall(toolCallId: "missing", toolName: "missing", input: "{}")
        let provider = ToolCall(
            toolCallId: "provider", toolName: "provider", input: "{}", providerExecuted: true
        )
        let terminalOne = ToolCall(toolCallId: "terminal-one", toolName: "one", input: "{}")
        let terminalTwo = ToolCall(toolCallId: "terminal-two", toolName: "two", input: "{}")
        let prompt: Prompt = [
            Message(
                role: .assistant,
                content: [.toolCall(first), .toolCall(missing), .toolCall(provider)],
                producer: destination
            ),
            .toolResult(toolCallId: first.toolCallId, toolName: first.toolName, result: "present"),
            .user("continue"),
            Message(
                role: .assistant,
                content: [.toolCall(terminalOne), .toolCall(terminalTwo)],
                producer: destination
            ),
        ]

        let transformed = transformer(for: destination).transform(prompt)
        let userIndex = try #require(transformed.firstIndex { $0.role == .user })
        let beforeUser = try #require(toolResult(in: transformed[userIndex - 1]))
        #expect(beforeUser.toolCallId == missing.toolCallId)
        #expect(beforeUser.result == Self.missingResultValue)
        #expect(beforeUser.content == [.text(ConversationTransformer.missingToolResult)])
        #expect(beforeUser.isError)

        let trailing = transformed.suffix(2).compactMap { message in
            message.content.compactMap {
                if case .toolResult(let result) = $0 { result } else { nil }
            }.first
        }
        #expect(trailing.map(\.toolCallId) == [terminalOne.toolCallId, terminalTwo.toolCallId])
        #expect(!transformed.contains { message in
            message.content.contains {
                if case .toolResult(let result) = $0 {
                    return result.toolCallId == provider.toolCallId
                }
                return false
            }
        })
    }

    @Test("failed and aborted turns are removed without classifying successful terminals as failures")
    func removesOnlyFailedAndAbortedTurns() throws {
        let destination = openAIResponses
        let prompt: Prompt = [
            .user("start"),
            Message(
                role: .assistant,
                content: [.text("partial failure"), .toolCall(ToolCall(
                    toolCallId: "failed-call", toolName: "bad", input: "{}"
                ))],
                producer: destination,
                outcome: .failed
            ),
            Message(
                role: .assistant,
                content: [.text("partial abort")],
                producer: destination,
                outcome: .aborted
            ),
            Message(role: .assistant, content: [.text("length")], producer: destination),
            Message(role: .assistant, content: [.text("filter")], producer: destination),
            Message(role: .assistant, content: [.text("tool")], producer: destination),
            Message(role: .assistant, content: [.text("future")], producer: destination),
        ]

        let transformed = transformer(for: destination).transform(prompt)
        #expect(transformed.compactMap { $0.role == .assistant ? $0.text : nil } == [
            "length", "filter", "tool", "future",
        ])
        #expect(toolResult(in: transformed) == nil)

        let failed = AIResponse(parts: [
            .finish(usage: .empty, finishReason: .init(unified: .error, raw: "failed"))
        ])
        let aborted = AIResponse(parts: [
            .finish(usage: .empty, finishReason: .init(unified: .error, raw: "cancelled"))
        ])
        let successfulReasons: [FinishReason] = [
            .init(unified: .length, raw: "max_tokens"),
            .init(unified: .contentFilter, raw: "safety"),
            .init(unified: .toolCalls, raw: "tool_use"),
            .init(unified: .other, raw: "future_success"),
        ]
        #expect(failed.assistantMessage.outcome == .failed)
        #expect(aborted.assistantMessage.outcome == .aborted)
        let persisted = try JSONDecoder().decode(
            Message.self,
            from: JSONEncoder().encode(aborted.assistantMessage)
        )
        #expect(persisted.outcome == .aborted)
        for reason in successfulReasons {
            #expect(AIResponse(parts: [.finish(
                usage: .empty, finishReason: reason
            )]).assistantMessage.outcome == nil)
        }
    }

    @Test("non-vision destinations coalesce exact user and tool placeholders in order")
    func downgradesAndCoalescesImages() throws {
        let destination = ModelDestination(
            providerId: "anthropic",
            apiId: WireProtocol.anthropicMessages.rawValue,
            modelId: "text-only"
        )
        let image = FilePart(mediaType: "image/png", data: .base64("image"))
        let pdf = FilePart(
            mediaType: "application/pdf", data: .base64("pdf"), filename: "document.pdf"
        )
        let call = ToolCall(toolCallId: "call", toolName: "vision", input: "{}")
        let prompt: Prompt = [
            Message(role: .user, content: [
                .text("before"), .file(image), .file(image),
                .text(ConversationTransformer.omittedUserImage), .file(image),
                .file(pdf), .file(image), .text("after"),
            ]),
            Message(role: .assistant, content: [.toolCall(call)], producer: destination),
            .toolResult(
                toolCallId: call.toolCallId,
                toolName: call.toolName,
                content: [
                    .text("tool-before"), .file(image), .file(image),
                    .text(ConversationTransformer.omittedToolImage), .file(image),
                    .file(pdf), .file(image), .text("tool-after"),
                ]
            ),
        ]

        let transformed = transformer(for: destination, supportsVision: false).transform(prompt)
        #expect(transformed[0].content == [
            .text("before"),
            .text(ConversationTransformer.omittedUserImage),
            .text(ConversationTransformer.omittedUserImage),
            .file(pdf),
            .text(ConversationTransformer.omittedUserImage),
            .text("after"),
        ])
        let tool = try #require(toolResult(in: transformed))
        #expect(tool.content == [
            .text("tool-before"),
            .text(ConversationTransformer.omittedToolImage),
            .text(ConversationTransformer.omittedToolImage),
            .file(pdf),
            .text(ConversationTransformer.omittedToolImage),
            .text("tool-after"),
        ])
        #expect(tool.result == .null)
    }

    @Test("vision destinations pass user and structured tool images through unchanged")
    func visionPassThrough() {
        let destination = openAIResponses
        let image = FilePart(mediaType: "image/jpeg", data: .url("https://example/image.jpg"))
        let call = ToolCall(toolCallId: "call", toolName: "vision", input: "{}")
        let prompt: Prompt = [
            Message(role: .user, content: [.text("look"), .file(image)]),
            Message(role: .assistant, content: [.toolCall(call)], producer: destination),
            .toolResult(
                toolCallId: call.toolCallId,
                toolName: call.toolName,
                content: [.text("found"), .file(image)]
            ),
        ]
        #expect(transformer(for: destination, supportsVision: true).transform(prompt) == prompt)
    }

    @Test("transformation is idempotent and leaves the caller's values unchanged")
    func idempotentAndNonMutating() {
        let source = openAIResponses
        let destination = ModelDestination(
            providerId: "anthropic",
            apiId: WireProtocol.anthropicMessages.rawValue,
            modelId: "text-only"
        )
        let image = FilePart(mediaType: "image/png", data: .base64("image"))
        let original: Prompt = [
            .user("start"),
            Message(role: .user, content: [.file(image), .file(image)]),
            Message(
                role: .assistant,
                content: [
                    .reasoning("readable", providerMetadata: ["openai": ["item": opaqueReasoningItem]]),
                    .toolCall(ToolCall(
                        toolCallId: "unsafe|+/=id", toolName: "lookup", input: "{}",
                        providerMetadata: ["openai": ["item": opaqueCallItem]]
                    )),
                ],
                providerOptions: ["openai": ["outputItems": [opaqueReasoningItem, opaqueCallItem]]],
                producer: source
            ),
        ]
        let snapshot = original
        let subject = transformer(for: destination, supportsVision: false)
        let once = subject.transform(original)
        let twice = subject.transform(once)

        #expect(original == snapshot)
        #expect(twice == once)
        #expect(once != original)
    }

    @Test("direct request preparation transforms once and isolates request options")
    func sharedPreparationBoundary() throws {
        let destination = ModelDestination(
            providerId: "target",
            apiId: WireProtocol.openAICompletions.rawValue,
            modelId: "target-model"
        )
        let model = ModelInfo(id: destination.modelId, modalities: .init(input: ["text"]))
        let provider = ProviderInfo(
            id: destination.providerId,
            api: "https://target.example",
            adapter: destination.apiId,
            models: [model]
        )
        let sourceMessage = Message(
            role: .assistant,
            content: [.reasoning("readable", providerMetadata: [
                "anthropic": ["signature": "private"]
            ])],
            producer: openAIResponses
        )
        let options = CallOptions(
            model: destination.modelId,
            prompt: [sourceMessage],
            providerOptions: [
                destination.providerId: ["target_only": true],
                "other": ["must_not_leak": true],
            ]
        )
        let client = AIClient(provider: provider, configuration: .init())

        let prepared = try client.prepare(options)

        #expect(options.prompt == [sourceMessage])
        #expect(prepared.options.prompt.first?.content == [.text("readable")])
        #expect(prepared.options.providerOptions == [
            destination.providerId: ["target_only": true]
        ])
        #expect(client.encode(prepared).body["target_only"] == true)
        #expect(client.encode(prepared).body["must_not_leak"] == nil)
    }

    private func transformer(
        for destination: ModelDestination,
        supportsVision: Bool = true
    ) -> ConversationTransformer {
        ConversationTransformer(
            destination: destination,
            model: ModelInfo(
                id: destination.modelId,
                modalities: .init(input: supportsVision ? ["text", "image"] : ["text"])
            )
        )
    }

    private func resultMessage(call: ToolCall, itemId: String) -> Message {
        Message(role: .tool, content: [.toolResult(ToolResult(
            toolCallId: call.toolCallId,
            toolName: call.toolName,
            result: "done",
            providerMetadata: ["openai": ["itemId": .string(itemId)]]
        ))])
    }

    private func text(in part: ContentPart) -> String? {
        if case .text(let text) = part { text } else { nil }
    }

    private func toolCall(in message: Message) -> ToolCall? {
        message.content.compactMap {
            if case .toolCall(let call) = $0 { call } else { nil }
        }.first
    }

    private func toolResult(in message: Message) -> ToolResult? {
        message.content.compactMap {
            if case .toolResult(let result) = $0 { result } else { nil }
        }.first
    }

    private func toolResult(in prompt: Prompt) -> ToolResult? {
        prompt.lazy.compactMap(toolResult(in:)).first
    }

    private static func isSafeID(_ id: String) -> Bool {
        !id.isEmpty && id.unicodeScalars.allSatisfy { scalar in
            let value = scalar.value
            return (65...90).contains(value) || (97...122).contains(value)
                || (48...57).contains(value) || value == 45 || value == 95
        }
    }

    private static let missingResultValue = JSONValue.string(
        ConversationTransformer.missingToolResult
    )
}
