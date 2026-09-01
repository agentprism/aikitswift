import Foundation
import Testing

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

@testable import AIKit

@Suite("Conversation transformation wire integration", .serialized)
struct ConversationTransformationIntegrationTests {
    @Test("facade switching transforms history before all five wire encoders")
    func facadeTransformsAllFiveOutgoingBodies() async throws {
        TransformStubProtocol.serve { request in
            switch request.url?.host() {
            case "responses.example":
                .json(Self.responsesResponse(model: "responses-model"))
            case "codex.example":
                .sse(Self.codexResponse(model: "codex-model"))
            case "anthropic.example":
                .json(Self.anthropicResponse(model: "anthropic-model"))
            case "chat.example":
                .json(Self.chatResponse(model: "chat-model"))
            case "google.example":
                .json(Self.googleResponse(model: "gemini-3-pro"))
            default:
                .json(#"{"error":"unexpected host"}"#, status: 500)
            }
        }

        let providers = Self.providers
        let facade = AIFacade(
            providers: providers,
            configurations: Dictionary(uniqueKeysWithValues: providers.map { provider in
                let authorization: AIClient.Authorization = provider.id == "openai-codex"
                    ? .oauth(OAuthCredential(accessToken: "stub-token", accountId: "account-1"))
                    : .apiKey("stub-key")
                return (provider.id, .init(authorization: authorization))
            }),
            session: Self.session()
        )
        let requestOptions = ProviderMetadata(uniqueKeysWithValues: providers.map {
            ($0.id, ["destination_marker": .string($0.id)])
        } + [("must-not-leak", ["foreign_option": true])])

        for provider in providers {
            let modelId = try #require(provider.models?.first?.id)
            let destination = try facade.destination(providerId: provider.id, modelId: modelId)
            _ = try await facade.generate(AIRequest(
                destination: destination,
                prompt: Self.foreignHistory,
                providerOptions: requestOptions
            ))
        }

        let requests = TransformStubProtocol.requests
        #expect(requests.count == 5)
        let bodies = try Dictionary(uniqueKeysWithValues: requests.map { request in
            let host = try #require(request.url?.host())
            let body = try JSONValue.decode(from: try #require(request.httpBody))
            return (host, body)
        })

        for provider in providers {
            let api = try #require(provider.api)
            let host = try #require(URL(string: api)?.host())
            let body = try #require(bodies[host])
            let encoded = try body.encodedString()
            #expect(body["destination_marker"]?.stringValue == provider.id)
            #expect(body["foreign_option"] == nil)
            #expect(encoded.contains("visible-reasoning"))
            #expect(!encoded.contains("encrypted-source-secret"))
            #expect(!encoded.contains("source_namespace"))
            #expect(!encoded.contains("call|unsafe+/=source"))
            #expect(encoded.contains("tool-before"))
            #expect(encoded.contains("tool-after"))
            #expect(encoded.contains(Self.toolImageData))
        }

        let responses = try #require(bodies["responses.example"])
        let codex = try #require(bodies["codex.example"])
        let anthropic = try #require(bodies["anthropic.example"])
        let chat = try #require(bodies["chat.example"])
        let google = try #require(bodies["google.example"])

        #expect(responses["input"]?.arrayValue != nil)
        #expect(responses["messages"] == nil)
        try expectResponsesCallAndResultLinked(responses)

        #expect(codex["input"]?.arrayValue != nil)
        #expect(codex["store"]?.boolValue == false)
        try expectResponsesCallAndResultLinked(codex)

        #expect(anthropic["messages"]?.arrayValue != nil)
        #expect(anthropic["max_tokens"] != nil)
        let anthropicBlocks = (anthropic["messages"]?.arrayValue ?? [])
            .flatMap { $0["content"]?.arrayValue ?? [] }
        let anthropicCall = try #require(anthropicBlocks.first {
            $0["type"]?.stringValue == "tool_use"
        })
        let anthropicResult = try #require(anthropicBlocks.first {
            $0["type"]?.stringValue == "tool_result"
        })
        #expect(anthropicCall["id"] == anthropicResult["tool_use_id"])
        #expect(anthropicResult["content"]?.arrayValue?.map {
            $0["type"]?.stringValue
        } == ["text", "image", "text"])
        #expect(anthropicResult["content"]?[1]?["source"]?["data"]?.stringValue == Self.toolImageData)

        #expect(chat["messages"]?.arrayValue != nil)
        #expect(chat["contents"] == nil)
        let chatMessages = chat["messages"]?.arrayValue ?? []
        let chatCallIds = chatMessages.compactMap {
            $0["tool_calls"]?[0]?["id"]?.stringValue
        }
        let chatCallId = try #require(chatCallIds.first)
        let chatResult = try #require(chatMessages.first {
            $0["role"]?.stringValue == "tool"
        })
        #expect(chatCallId == chatResult["tool_call_id"]?.stringValue)
        #expect(chatCallId.count <= 40)
        #expect(chatResult["content"]?.stringValue == "tool-before\ntool-after")
        let chatImageTurn = try #require(chatMessages.first { message in
            message["content"]?.arrayValue?.contains {
                $0["image_url"] != nil
            } == true
        })
        #expect(chatImageTurn["content"]?[0]?["text"]?.stringValue
            == "Attached image(s) from tool result:")
        #expect(chatImageTurn["content"]?[1]?["image_url"]?["url"]?.stringValue
            == "data:image/png;base64,\(Self.toolImageData)")

        #expect(google["contents"]?.arrayValue != nil)
        #expect(google["messages"] == nil)
        let googleParts = (google["contents"]?.arrayValue ?? [])
            .flatMap { $0["parts"]?.arrayValue ?? [] }
        let googleCallIds = googleParts.compactMap {
            $0["functionCall"]?["id"]?.stringValue
        }
        let googleResultIds = googleParts.compactMap {
            $0["functionResponse"]?["id"]?.stringValue
        }
        let googleCallId = try #require(googleCallIds.first)
        let googleResultId = try #require(googleResultIds.first)
        #expect(googleCallId == googleResultId)
        #expect(googleCallId.count <= 64)
        let googleResult = try #require(googleParts.first {
            $0["functionResponse"] != nil
        }?["functionResponse"])
        #expect(googleResult["response"]?["output"]?.stringValue == "tool-before\ntool-after")
        #expect(googleResult["parts"]?[0]?["inlineData"]?["data"]?.stringValue
            == Self.toolImageData)
    }

    @Test("text-only Chat and Google bodies contain placeholders without image data")
    func unsupportedToolImagesReachChatAndGoogleBodies() async throws {
        TransformStubProtocol.serve { request in
            switch request.url?.host() {
            case "chat-text.example":
                .json(Self.chatResponse(model: "chat-text-model"))
            case "google-text.example":
                .json(Self.googleResponse(model: "gemini-3-text"))
            default:
                .json(#"{"error":"unexpected host"}"#, status: 500)
            }
        }

        let providers = [
            Self.provider(
                id: "chat-text", host: "chat-text.example", wire: .openAICompletions,
                modelId: "chat-text-model", supportsVision: false
            ),
            Self.provider(
                id: "google-text", host: "google-text.example", wire: .googleGenerativeAI,
                modelId: "gemini-3-text", supportsVision: false
            ),
        ]
        let facade = AIFacade(
            providers: providers,
            configurations: Dictionary(uniqueKeysWithValues: providers.map {
                ($0.id, .init(apiKey: "stub-key"))
            }),
            session: Self.session()
        )
        let image = FilePart(mediaType: "image/png", data: .base64(Self.unsupportedImageData))
        let call = ToolCall(toolCallId: "call-with-images", toolName: "inspect", input: "{}")
        let prompt: Prompt = [
            Message(role: .user, content: [
                .text("user-before"), .file(image), .file(image), .text("user-after"),
            ]),
            Message(role: .assistant, content: [.toolCall(call)]),
            .toolResult(
                toolCallId: call.toolCallId,
                toolName: call.toolName,
                content: [
                    .text("tool-before"), .file(image), .file(image), .text("tool-after"),
                ]
            ),
        ]

        for provider in providers {
            let destination = try facade.destination(
                providerId: provider.id,
                modelId: try #require(provider.models?.first?.id)
            )
            _ = try await facade.generate(AIRequest(destination: destination, prompt: prompt))
        }

        let bodies = try Dictionary(uniqueKeysWithValues: TransformStubProtocol.requests.map {
            let host = try #require($0.url?.host())
            return (host, try JSONValue.decode(from: try #require($0.httpBody)))
        })
        for body in bodies.values {
            let encoded = try body.encodedString()
            #expect(encoded.contains("(image omitted: model does not support images)"))
            #expect(encoded.contains("(tool image omitted: model does not support images)"))
            #expect(!encoded.contains(Self.unsupportedImageData))
            #expect(!encoded.contains("data:image"))
            #expect(!encoded.contains("inlineData"))
        }

        let chatMessages = try #require(bodies["chat-text.example"]?["messages"]?.arrayValue)
        let chatTool = try #require(chatMessages.first {
            $0["role"]?.stringValue == "tool"
        })
        #expect(chatTool["content"]?.stringValue == [
            "tool-before",
            "(tool image omitted: model does not support images)",
            "tool-after",
        ].joined(separator: "\n"))

        let googleParts = try #require(bodies["google-text.example"]?["contents"]?.arrayValue)
            .flatMap { $0["parts"]?.arrayValue ?? [] }
        let googleResponse = try #require(googleParts.first {
            $0["functionResponse"] != nil
        }?["functionResponse"]?["response"]?["output"]?.stringValue)
        #expect(googleResponse == [
            "tool-before",
            "(tool image omitted: model does not support images)",
            "tool-after",
        ].joined(separator: "\n"))
    }

    @Test("direct AIClient stream and complete paths encode identical transformed history")
    func directStreamAndGenerateStayInParity() async throws {
        TransformStubProtocol.serve { request in
            if request.value(forHTTPHeaderField: "accept") == "text/event-stream" {
                return .sse(Self.chatStreamResponse(model: "chat-model"))
            }
            return .json(Self.chatResponse(model: "chat-model"))
        }
        let provider = try #require(Self.providers.first { $0.id == "chat-target" })
        let client = AIClient(
            provider: provider,
            configuration: .init(apiKey: "stub-key"),
            session: Self.session()
        )
        let options = CallOptions(model: "chat-model", prompt: Self.foreignHistory)
        let snapshot = options.prompt

        _ = try await client.generate(options)
        _ = try await client.stream(options).collect()

        let bodies = try TransformStubProtocol.requests.map {
            try JSONValue.decode(from: try #require($0.httpBody))
        }
        #expect(bodies.count == 2)
        #expect(bodies[0]["messages"] == bodies[1]["messages"])
        #expect(try bodies[0]["messages"]?.encodedString().contains("visible-reasoning") == true)
        #expect(try bodies[0]["messages"]?.encodedString().contains("encrypted-source-secret") == false)
        #expect(options.prompt == snapshot)
    }

    private static let sourceDestination = ModelDestination(
        providerId: "source-provider",
        apiId: WireProtocol.openAIResponses.rawValue,
        modelId: "source-model"
    )

    private static let sourceItem: JSONValue = [
        "type": "function_call",
        "id": "fc/source+/=item",
        "call_id": "call|unsafe+/=source",
        "name": "lookup",
        "namespace": "source_namespace",
        "arguments": "{}",
        "status": "completed",
    ]

    private static let toolImageData = "supported-tool-image-bytes"
    private static let unsupportedImageData = "unsupported-tool-image-bytes"

    private static let foreignHistory: Prompt = {
        let call = ToolCall(
            toolCallId: "call|unsafe+/=source",
            toolName: "lookup",
            namespace: "source_namespace",
            input: "{}",
            providerMetadata: [
                "openai": ["item": sourceItem],
                "google": ["thoughtSignature": "encrypted-source-secret"],
            ]
        )
        return [
            .user("original-user"),
            Message(
                role: .assistant,
                content: [
                    .reasoning("visible-reasoning", providerMetadata: [
                        "openai": [
                            "item": [
                                "type": "reasoning",
                                "id": "rs_source",
                                "encrypted_content": "encrypted-source-secret",
                            ]
                        ]
                    ]),
                    .toolCall(call),
                    .text("assistant-text"),
                ],
                providerOptions: [
                    "openai": ["outputItems": [
                        [
                            "type": "reasoning", "id": "rs_source",
                            "encrypted_content": "encrypted-source-secret",
                        ],
                        sourceItem,
                    ]]
                ],
                producer: sourceDestination
            ),
            Message(role: .tool, content: [.toolResult(ToolResult(
                toolCallId: call.toolCallId,
                toolName: call.toolName,
                result: .null,
                content: [
                    .text("tool-before"),
                    .file(FilePart(mediaType: "image/png", data: .base64(toolImageData))),
                    .text("tool-after"),
                ],
                providerMetadata: ["openai": [
                    "itemId": "foreign-result-id",
                    "output": ["secret": "encrypted-source-secret"],
                ]]
            ))]),
            .user("continue"),
        ]
    }()

    private static let providers: [ProviderInfo] = [
        provider(
            id: "openai", host: "responses.example", wire: .openAIResponses,
            modelId: "responses-model"
        ),
        provider(
            id: "openai-codex", host: "codex.example", wire: .openAICodex,
            modelId: "codex-model"
        ),
        provider(
            id: "anthropic", host: "anthropic.example", wire: .anthropicMessages,
            modelId: "anthropic-model"
        ),
        provider(
            id: "chat-target", host: "chat.example", wire: .openAICompletions,
            modelId: "chat-model"
        ),
        provider(
            id: "google", host: "google.example", wire: .googleGenerativeAI,
            modelId: "gemini-3-pro"
        ),
    ]

    private static func provider(
        id: String,
        host: String,
        wire: WireProtocol,
        modelId: String,
        supportsVision: Bool = true
    ) -> ProviderInfo {
        ProviderInfo(
            id: id,
            api: "https://\(host)",
            adapter: wire.rawValue,
            models: [ModelInfo(
                id: modelId,
                reasoning: .init(supported: true),
                toolCall: true,
                modalities: .init(
                    input: supportsVision ? ["text", "image"] : ["text"],
                    output: ["text"]
                )
            )]
        )
    }

    private static func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [TransformStubProtocol.self]
        return URLSession(configuration: configuration)
    }

    private func expectResponsesCallAndResultLinked(_ body: JSONValue) throws {
        let input = body["input"]?.arrayValue ?? []
        let call = try #require(input.first {
            $0["type"]?.stringValue == "function_call"
        })
        let toolResult = try #require(input.first {
            $0["type"]?.stringValue == "function_call_output"
        })
        #expect(call["call_id"] == toolResult["call_id"])
        #expect(call["id"]?.stringValue?.hasPrefix("fc_") == true)
        #expect(call["id"]?.stringValue != Self.sourceItem["id"]?.stringValue)
        #expect(toolResult["output"]?.arrayValue?.map {
            $0["type"]?.stringValue
        } == ["input_text", "input_image", "input_text"])
        #expect(toolResult["output"]?[1]?["image_url"]?.stringValue
            == "data:image/png;base64,\(Self.toolImageData)")
    }

    private static func responsesResponse(model: String) -> String {
        """
        {"id":"resp_complete","status":"completed","model":"\(model)","output":[],
         "usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}
        """
    }

    private static func codexResponse(model: String) -> String {
        """
        data: {"type":"response.done","sequence_number":1,"response":{"id":"resp_codex","status":"completed","model":"\(model)","output":[],"usage":{"input_tokens":2,"output_tokens":1,"total_tokens":3}}}


        """
    }

    private static func anthropicResponse(model: String) -> String {
        """
        {"type":"message","id":"msg_complete","model":"\(model)","role":"assistant",
         "content":[{"type":"text","text":"ok"}],"stop_reason":"end_turn",
         "usage":{"input_tokens":2,"output_tokens":1}}
        """
    }

    private static func chatResponse(model: String) -> String {
        """
        {"id":"chat_complete","model":"\(model)","created":1,
         "choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}],
         "usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}
        """
    }

    private static func chatStreamResponse(model: String) -> String {
        """
        data: {"id":"chat_stream","model":"\(model)","created":1,"choices":[{"index":0,"delta":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}

        data: [DONE]


        """
    }

    private static func googleResponse(model: String) -> String {
        """
        {"responseId":"google_complete","modelVersion":"\(model)",
         "candidates":[{"index":0,"content":{"role":"model","parts":[{"text":"ok"}]},"finishReason":"STOP"}],
         "usageMetadata":{"promptTokenCount":2,"candidatesTokenCount":1,"totalTokenCount":3}}
        """
    }
}

private struct TransformStubResponse: Sendable {
    var status: Int
    var contentType: String
    var body: Data

    static func json(_ body: String, status: Int = 200) -> Self {
        Self(status: status, contentType: "application/json", body: Data(body.utf8))
    }

    static func sse(_ body: String, status: Int = 200) -> Self {
        Self(status: status, contentType: "text/event-stream", body: Data(body.utf8))
    }
}

private final class TransformStubProtocol: URLProtocol, @unchecked Sendable {
    typealias Handler = @Sendable (URLRequest) -> TransformStubResponse

    nonisolated(unsafe) private static var handler: Handler = { _ in
        .json(#"{"error":"no stub configured"}"#, status: 500)
    }
    nonisolated(unsafe) private static var captured: [URLRequest] = []
    private static let lock = NSLock()

    static func serve(_ handler: @escaping Handler) {
        lock.lock()
        defer { lock.unlock() }
        Self.handler = handler
        captured = []
    }

    static var requests: [URLRequest] {
        lock.lock()
        defer { lock.unlock() }
        return captured
    }

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        var capturedRequest = request
        if capturedRequest.httpBody == nil, let stream = request.httpBodyStream {
            stream.open()
            defer { stream.close() }
            var body = Data()
            let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: 4_096)
            defer { buffer.deallocate() }
            while stream.hasBytesAvailable {
                let count = stream.read(buffer, maxLength: 4_096)
                guard count > 0 else { break }
                body.append(buffer, count: count)
            }
            capturedRequest.httpBody = body
        }

        Self.lock.lock()
        Self.captured.append(capturedRequest)
        let handler = Self.handler
        Self.lock.unlock()

        let stub = handler(capturedRequest)
        let response = HTTPURLResponse(
            url: request.url!,
            statusCode: stub.status,
            httpVersion: "HTTP/1.1",
            headerFields: ["Content-Type": stub.contentType]
        )!
        client?.urlProtocol(self, didReceive: response, cacheStoragePolicy: .notAllowed)
        client?.urlProtocol(self, didLoad: stub.body)
        client?.urlProtocolDidFinishLoading(self)
    }

    override func stopLoading() {}
}
