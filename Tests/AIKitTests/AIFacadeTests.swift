import Foundation
import Testing

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

@testable import AIKit

@Suite("Provider-neutral facade", .serialized)
struct AIFacadeTests {
    private let sharedModel = ModelInfo(
        id: "shared-model",
        reasoning: .init(supported: true),
        toolCall: true,
        modalities: .init(input: ["text", "image"], output: ["text"])
    )

    private var alpha: ProviderInfo {
        ProviderInfo(
            id: "alpha",
            api: "https://alpha.example",
            adapter: WireProtocol.openAICompletions.rawValue,
            models: [sharedModel]
        )
    }

    private var beta: ProviderInfo {
        ProviderInfo(
            id: "beta",
            api: "https://beta.example",
            adapter: WireProtocol.anthropicMessages.rawValue,
            models: [sharedModel]
        )
    }

    private func session() -> URLSession {
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [FacadeStubProtocol.self]
        return URLSession(configuration: configuration)
    }

    private func facade(providers: [ProviderInfo]? = nil) -> AIFacade {
        AIFacade(
            providers: providers ?? [alpha, beta],
            configurations: [
                "alpha": .init(
                    apiKey: "alpha-key",
                    extraHeaders: ["x-provider-config": "alpha"]
                ),
                "beta": .init(
                    apiKey: "beta-key",
                    extraHeaders: ["x-provider-config": "beta"]
                ),
            ],
            session: session()
        )
    }

    @Test("provider, API, and model all participate in destination identity")
    func destinationIdentityIsComplete() {
        let alpha = ModelDestination(providerId: "alpha", apiId: "openai", modelId: "same")
        let beta = ModelDestination(providerId: "beta", apiId: "openai", modelId: "same")
        let responses = ModelDestination(
            providerId: "alpha",
            apiId: "openai-responses",
            modelId: "same"
        )

        #expect(alpha != beta)
        #expect(alpha != responses)
        #expect(Set([alpha, beta, responses]).count == 3)
    }

    @Test("catalog and custom providers derive API identity from their declaration")
    func selectsCatalogAndCustomModels() throws {
        let catalog = AIFacade(configurations: [:])
        let catalogDestination = try catalog.destination(
            providerId: "anthropic",
            modelId: "claude-haiku-4-5"
        )
        #expect(catalogDestination.apiId == WireProtocol.anthropicMessages.rawValue)
        #expect(try catalog.model(for: catalogDestination).supportsTools)

        let custom = facade(providers: [alpha])
        let customDestination = try custom.destination(
            providerId: "alpha",
            modelId: "shared-model"
        )
        #expect(customDestination == ModelDestination(
            providerId: "alpha",
            apiId: WireProtocol.openAICompletions.rawValue,
            modelId: "shared-model"
        ))
        #expect(try custom.model(for: customDestination).supportsVision)
    }

    @Test("one facade switches providers, wires, credentials, and options per call")
    func switchesMockedProviders() async throws {
        FacadeStubProtocol.serve { request in
            switch request.url?.host() {
            case "alpha.example":
                .json(Self.openAIResponse(text: "from alpha", model: "alpha-actual"))
            case "beta.example":
                .json(Self.anthropicResponse(text: "from beta", model: "beta-actual"))
            default:
                .json(#"{"error":"unexpected host"}"#, status: 500)
            }
        }

        let facade = facade()
        let alphaDestination = try facade.destination(providerId: "alpha", modelId: "shared-model")
        let betaDestination = try facade.destination(providerId: "beta", modelId: "shared-model")
        let providerOptions: ProviderMetadata = [
            "alpha": [
                "alpha_only": true,
                "collision": "alpha-value",
            ],
            "beta": [
                "beta_only": true,
                "collision": "beta-value",
            ],
        ]

        let alphaResponse = try await facade.generate(AIRequest(
            destination: alphaDestination,
            prompt: [.user("hi")],
            providerOptions: providerOptions
        ))
        let betaResponse = try await facade.generate(AIRequest(
            destination: betaDestination,
            prompt: [.user("hi")],
            providerOptions: providerOptions
        ))

        #expect(alphaResponse.text == "from alpha")
        #expect(betaResponse.text == "from beta")
        #expect(alphaResponse.assistantMessage.producer == alphaDestination)
        #expect(betaResponse.assistantMessage.producer == betaDestination)
        #expect(alphaResponse.assistantMessage.responseModelId == "alpha-actual")
        #expect(betaResponse.assistantMessage.responseModelId == "beta-actual")

        let requests = FacadeStubProtocol.requests
        let alphaRequest = try #require(requests.first { $0.url?.host() == "alpha.example" })
        let betaRequest = try #require(requests.first { $0.url?.host() == "beta.example" })
        let alphaBody = try JSONValue.decode(from: try #require(alphaRequest.httpBody))
        let betaBody = try JSONValue.decode(from: try #require(betaRequest.httpBody))

        #expect(alphaRequest.url?.path == "/v1/chat/completions")
        #expect(betaRequest.url?.path == "/v1/messages")
        #expect(alphaRequest.value(forHTTPHeaderField: "authorization") == "Bearer alpha-key")
        #expect(betaRequest.value(forHTTPHeaderField: "x-api-key") == "beta-key")
        #expect(alphaRequest.value(forHTTPHeaderField: "x-provider-config") == "alpha")
        #expect(betaRequest.value(forHTTPHeaderField: "x-provider-config") == "beta")
        #expect(alphaBody["collision"] == "alpha-value")
        #expect(betaBody["collision"] == "beta-value")
        #expect(alphaBody["alpha_only"] == true)
        #expect(alphaBody["beta_only"] == nil)
        #expect(betaBody["beta_only"] == true)
        #expect(betaBody["alpha_only"] == nil)
    }

    @Test("one preparation boundary resolves immutable identity and capabilities")
    func preparesDestinationAndCapabilitiesOnce() throws {
        let client = AIClient(
            provider: alpha,
            configuration: .init(apiKey: "alpha-key"),
            session: session()
        )
        let options = CallOptions(model: sharedModel.id, prompt: [.user("hi")])

        let prepared = try client.prepare(options)

        #expect(prepared.destination == ModelDestination(
            providerId: alpha.id,
            apiId: WireProtocol.openAICompletions.rawValue,
            modelId: sharedModel.id
        ))
        #expect(prepared.provider == alpha)
        #expect(prepared.model == sharedModel)
        #expect(prepared.wire == .openAICompletions)
        #expect(prepared.options.model == sharedModel.id)
    }

    @Test("stream collection and complete generation preserve the same identity and outcome")
    func streamAndCompleteAreInParity() async throws {
        FacadeStubProtocol.serve { request in
            if request.value(forHTTPHeaderField: "accept") == "text/event-stream" {
                return .sse(Self.openAIStream(text: "same answer", model: "routed-model"))
            }
            return .json(Self.openAIResponse(text: "same answer", model: "routed-model"))
        }

        let facade = facade(providers: [alpha])
        let destination = try facade.destination(providerId: "alpha", modelId: "shared-model")
        let request = AIRequest(destination: destination, prompt: [.user("hi")])

        let streamed = try await facade.stream(request).collect()
        let completed = try await facade.generate(request)

        #expect(streamed.text == completed.text)
        #expect(streamed.usage == completed.usage)
        #expect(streamed.finishReason == completed.finishReason)
        #expect(streamed.assistantMessage.producer == destination)
        #expect(completed.assistantMessage.producer == destination)
        #expect(streamed.assistantMessage.responseModelId == "routed-model")
        #expect(completed.assistantMessage.responseModelId == "routed-model")
        #expect(streamed.assistantMessage.producer?.modelId == "shared-model")
    }

    @Test("a supplied actual response model does not depend on a response id")
    func actualResponseModelDoesNotRequireResponseId() async throws {
        FacadeStubProtocol.serve { _ in
            .json("""
                {"model":"server-fallback","created":1,
                 "choices":[{"index":0,"message":{"role":"assistant","content":"fallback"},"finish_reason":"stop"}]}
                """)
        }

        let facade = facade(providers: [alpha])
        let destination = try facade.destination(providerId: "alpha", modelId: "shared-model")
        let response = try await facade.generate(AIRequest(
            destination: destination,
            prompt: [.user("hi")]
        ))

        #expect(response.assistantMessage.producer == destination)
        #expect(response.assistantMessage.responseModelId == "server-fallback")
    }

    @Test("requested identity survives when the provider supplies no response metadata")
    func requestedIdentityDoesNotRequireResponseMetadata() async throws {
        FacadeStubProtocol.serve { _ in
            .json("""
                {"choices":[{"index":0,"message":{"role":"assistant","content":"plain"},"finish_reason":"stop"}]}
                """)
        }

        let facade = facade(providers: [alpha])
        let destination = try facade.destination(providerId: "alpha", modelId: "shared-model")
        let response = try await facade.generate(AIRequest(
            destination: destination,
            prompt: [.user("hi")]
        ))

        #expect(response.assistantMessage.producer == destination)
        #expect(response.assistantMessage.responseModelId == nil)
    }

    @Test("direct AIClient remains available and records requested identity")
    func directClientStillWorks() async throws {
        FacadeStubProtocol.serve { _ in
            .json(Self.openAIResponse(text: "direct", model: "server-model"))
        }
        let client = AIClient(
            provider: alpha,
            configuration: .init(apiKey: "direct-key"),
            session: session()
        )

        let response = try await client.generate(CallOptions(
            model: "shared-model",
            prompt: [.user("hi")]
        ))

        #expect(response.text == "direct")
        #expect(response.assistantMessage.producer == ModelDestination(
            providerId: "alpha",
            apiId: WireProtocol.openAICompletions.rawValue,
            modelId: "shared-model"
        ))
        #expect(response.assistantMessage.responseModelId == "server-model")
    }

    @Test("resolution errors identify the missing or conflicting component")
    func errorsAreExplicit() throws {
        let normal = alpha
        let missingAPI = ProviderInfo(
            id: "missing-api",
            api: "https://missing.example",
            models: [sharedModel]
        )
        let unsupported = ProviderInfo(
            id: "unsupported",
            api: "https://unsupported.example",
            adapter: "future-protocol",
            models: [sharedModel]
        )
        let facade = AIFacade(
            providers: [normal, missingAPI, unsupported],
            configurations: ["alpha": .init()]
        )

        try expectError(.unknownProvider("unknown")) {
            _ = try facade.destination(providerId: "unknown", modelId: "shared-model")
        }
        try expectError(.missingModel(providerId: "alpha", modelId: "missing")) {
            _ = try facade.destination(providerId: "alpha", modelId: "missing")
        }
        try expectError(.missingAPI("missing-api")) {
            _ = try facade.destination(providerId: "missing-api", modelId: "shared-model")
        }
        try expectError(.unsupportedProtocol("future-protocol")) {
            _ = try facade.destination(providerId: "unsupported", modelId: "shared-model")
        }
        try expectError(.apiMismatch(
            providerId: "alpha",
            requested: WireProtocol.openAIResponses.rawValue,
            declared: WireProtocol.openAICompletions.rawValue
        )) {
            _ = try facade.model(for: ModelDestination(
                providerId: "alpha",
                apiId: WireProtocol.openAIResponses.rawValue,
                modelId: "shared-model"
            ))
        }

        let noConfiguration = AIFacade(providers: [normal], configurations: [:])
        let destination = try noConfiguration.destination(
            providerId: "alpha",
            modelId: "shared-model"
        )
        try expectError(.missingConfiguration("alpha")) {
            _ = try noConfiguration.stream(AIRequest(
                destination: destination,
                prompt: [.user("hi")]
            ))
        }
    }

    @Test("hand-built assistant history can omit producer identity")
    func handBuiltHistoryRemainsCompatible() throws {
        let message = Message.assistant("hand built")
        #expect(message.producer == nil)
        #expect(message.responseModelId == nil)

        let encoded = try JSONEncoder().encode(message)
        let decoded = try JSONDecoder().decode(Message.self, from: encoded)
        #expect(decoded == message)
    }

    @Test("producer and actual model identity survive history persistence")
    func assistantIdentityIsCodable() throws {
        let producer = ModelDestination(
            providerId: "alpha",
            apiId: WireProtocol.openAICompletions.rawValue,
            modelId: "requested-model"
        )
        let message = Message(
            role: .assistant,
            content: [.text("persisted")],
            producer: producer,
            responseModelId: "actual-model"
        )

        let encoded = try JSONEncoder().encode(message)
        let decoded = try JSONDecoder().decode(Message.self, from: encoded)

        #expect(decoded == message)
        #expect(decoded.producer == producer)
        #expect(decoded.responseModelId == "actual-model")
    }

    private func expectError(
        _ expected: AIFacadeError.Kind,
        operation: () throws -> Void
    ) throws {
        do {
            try operation()
            Issue.record("Expected facade error \(expected)")
        } catch let error as AIFacadeError {
            #expect(error.kind == expected)
            #expect(!error.description.isEmpty)
        }
    }

    private static func openAIResponse(text: String, model: String) -> String {
        """
        {"id":"chatcmpl_complete","model":"\(model)","created":1,
         "choices":[{"index":0,"message":{"role":"assistant","content":"\(text)"},"finish_reason":"stop"}],
         "usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}
        """
    }

    private static func anthropicResponse(text: String, model: String) -> String {
        """
        {"type":"message","id":"msg_complete","model":"\(model)","role":"assistant",
         "content":[{"type":"text","text":"\(text)"}],"stop_reason":"end_turn",
         "usage":{"input_tokens":2,"output_tokens":2}}
        """
    }

    private static func openAIStream(text: String, model: String) -> String {
        """
        data: {"id":"chatcmpl_stream","created":1,"choices":[{"index":0,"delta":{"role":"assistant","content":"\(text)"},"finish_reason":"stop"}]}

        data: {"model":"\(model)","choices":[],"usage":{"prompt_tokens":2,"completion_tokens":2,"total_tokens":4}}

        data: [DONE]


        """
    }
}

private struct FacadeStubResponse: Sendable {
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

private final class FacadeStubProtocol: URLProtocol, @unchecked Sendable {
    typealias Handler = @Sendable (URLRequest) -> FacadeStubResponse

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
