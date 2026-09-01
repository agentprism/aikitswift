import Foundation

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Something went wrong before the stream produced any content.
public struct AIClientError: Error, Sendable, CustomStringConvertible {
    public enum Kind: Sendable {
        /// The provider is not in the catalog.
        case unknownProvider(String)
        /// The provider's adapter has no implementation here.
        case unsupportedProtocol(String)
        /// No base URL, from the catalog or the configuration.
        case missingBaseURL(String)
        /// The provider returned a non-2xx status.
        case http(status: Int, body: String)
        /// Codex requires both bearer authorization and ChatGPT account identity.
        case missingCodexCredential
        case missingCodexAccountId
        /// ChatGPT Codex has no supported live model-list endpoint in AIKit.
        case unsupportedModelListing(String)
    }

    public var kind: Kind

    public var description: String {
        switch kind {
        case .unknownProvider(let id):
            "Unknown provider '\(id)'. Check ProviderCatalog.all for available ids."
        case .unsupportedProtocol(let adapter):
            "Provider speaks '\(adapter)', which AIKit does not implement yet."
        case .missingBaseURL(let id):
            "Provider '\(id)' has no base URL; supply one in the configuration."
        case .http(let status, let body):
            "Provider returned HTTP \(status): \(body)"
        case .missingCodexCredential:
            "OpenAI Codex requires an OAuth bearer credential."
        case .missingCodexAccountId:
            "OpenAI Codex requires ChatGPT account identity in its OAuth access token."
        case .unsupportedModelListing(let protocolName):
            "Live model listing is not supported for '\(protocolName)'; use the bundled catalog."
        }
    }
}

// Without this, `error.localizedDescription` — the accessor every app reaches
// for first — shows Foundation's generic text instead of the message above.
extension AIClientError: LocalizedError {
    public var errorDescription: String? { description }
}

/// The single request-preparation result shared by stream and complete paths.
///
/// Priority-5 history transformation can consume this boundary without
/// re-resolving provider identity, API identity, or model capabilities.
struct PreparedAIRequest: Sendable {
    let destination: ModelDestination
    let provider: ProviderInfo
    let model: ModelInfo?
    let wire: WireProtocol
    let options: CallOptions

    fileprivate init(
        destination: ModelDestination,
        provider: ProviderInfo,
        model: ModelInfo?,
        wire: WireProtocol,
        options: CallOptions
    ) {
        self.destination = destination
        self.provider = provider
        self.model = model
        self.wire = wire
        self.options = options
    }
}

/// Sends a request to a provider and returns a normalized event stream.
///
/// The client is thin on purpose. It resolves which protocol a provider speaks,
/// encodes the request, opens the connection, and feeds bytes through the
/// matching mapper. Everything interesting lives in the wire layer; this is the
/// plumbing that connects it to a socket.
public struct AIClient: Sendable {

    /// How requests are authenticated.
    public enum Authorization: Sendable {
        case none
        case apiKey(String)
        /// A subscription or account login rather than a billed API key —
        /// Claude Pro/Max, OpenAI Codex, OpenRouter. These travel as bearer
        /// tokens on a different header than an API key, and some providers
        /// additionally gate them behind an opt-in.
        case oauth(OAuthCredential)
    }

    public struct Configuration: Sendable {
        public var authorization: Authorization
        /// Overrides the catalog's base URL. Required for providers that have
        /// none — self-hosted gateways, local runtimes.
        public var baseURL: URL?
        /// Merged into every request, and applied last so they can override.
        public var extraHeaders: [String: String]
        /// Optional persisted/refreshing OAuth source. Codex uses this before a
        /// request and once more after a 401; static `.oauth` authorization
        /// remains available for providers that do not need managed refresh.
        public var oauthCredentialProvider: (any OAuthCredentialProviding)?

        public init(
            authorization: Authorization = .none,
            baseURL: URL? = nil,
            extraHeaders: [String: String] = [:],
            oauthCredentialProvider: (any OAuthCredentialProviding)? = nil
        ) {
            self.authorization = authorization
            self.baseURL = baseURL
            self.extraHeaders = extraHeaders
            self.oauthCredentialProvider = oauthCredentialProvider
        }

        public init(
            apiKey: String?,
            baseURL: URL? = nil,
            extraHeaders: [String: String] = [:],
            oauthCredentialProvider: (any OAuthCredentialProviding)? = nil
        ) {
            self.init(
                authorization: apiKey.map(Authorization.apiKey) ?? .none,
                baseURL: baseURL,
                extraHeaders: extraHeaders,
                oauthCredentialProvider: oauthCredentialProvider
            )
        }
    }

    public var provider: ProviderInfo
    public var configuration: Configuration
    public var session: URLSession

    public init(provider: ProviderInfo, configuration: Configuration, session: URLSession = .shared) {
        self.provider = provider
        self.configuration = configuration
        self.session = session
    }

    /// Looks the provider up in the bundled catalog.
    public init(
        providerId: String,
        configuration: Configuration,
        session: URLSession = .shared
    ) throws {
        guard let provider = ProviderCatalog.provider(providerId) else {
            throw AIClientError(kind: .unknownProvider(providerId))
        }
        self.init(provider: provider, configuration: configuration, session: session)
    }

    // MARK: - Streaming

    /// Sends a request and yields normalized events as they arrive.
    public func stream(_ options: CallOptions) throws -> AsyncThrowingStream<StreamPart, any Error> {
        try stream(prepare(options))
    }

    func stream(_ prepared: PreparedAIRequest) throws -> AsyncThrowingStream<StreamPart, any Error> {
        let wire = prepared.wire
        let options = prepared.options
        let encoded = encode(prepared)
        let session = self.session
        let usesManagedCodexCredential = wire == .openAICodex
            && configuration.oauthCredentialProvider != nil
        // The explicit credential argument is a Codex request-time override.
        // Other wires must inspect the original authorization case so API keys
        // retain provider-specific headers instead of becoming bearer tokens.
        let initialCredential = wire == .openAICodex && !usesManagedCodexCredential
            ? configuredOAuthCredential
            : nil
        // Preserve the existing synchronous validation contract for every
        // ordinary provider (and static Codex credentials). Only managed Codex
        // auth requires request construction to wait on async credential I/O.
        let initialRequest = usesManagedCodexCredential ? nil : try makeRequest(
            wire: wire,
            options: options,
            body: encoded.body,
            oauthCredential: initialCredential
        )

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    // Warnings come from encoding, so the client emits
                    // `stream-start` itself and suppresses the mapper's. The
                    // contract stays: exactly one, first, carrying warnings.
                    continuation.yield(.streamStart(warnings: encoded.warnings))
                    var attachedProducer = false

                    var credential = initialCredential
                    var preparedRequest = initialRequest
                    if usesManagedCodexCredential {
                        credential = try await credentialForRequest(wire: wire)
                    }
                    var replayedAuthentication = false

                    requestLoop: while true {
                        try Task.checkCancellation()
                        let request: URLRequest
                        if let ready = preparedRequest {
                            request = ready
                        } else {
                            request = try makeRequest(
                                wire: wire,
                                options: options,
                                body: encoded.body,
                                oauthCredential: credential
                            )
                        }
                        preparedRequest = nil
                        let (bytes, response) = try await session.bytes(for: request)

                        if let http = response as? HTTPURLResponse,
                           !(200..<300).contains(http.statusCode) {
                            // The body holds the provider's error detail, which
                            // is collected verbatim before a retry or error.
                            var detail: [UInt8] = []
                            for try await byte in bytes { detail.append(byte) }

                            if wire == .openAICodex,
                               http.statusCode == 401,
                               !replayedAuthentication,
                               let rejected = credential?.accessToken,
                               let provider = configuration.oauthCredentialProvider {
                                replayedAuthentication = true
                                credential = try await provider.credential(afterRejecting: rejected)
                                continue requestLoop
                            }

                            throw AIClientError(kind: .http(
                                status: http.statusCode,
                                body: String(decoding: detail, as: UTF8.self)
                            ))
                        }

                        var mapper = wire.makeMapper()

                        let eventSource = SSEEventSource(bytes: bytes)
                        for try await event in eventSource.events {
                            for part in mapper.map(rawJSON: event.data) {
                                if case .streamStart = part { continue }
                                for identified in parts(
                                    attaching: prepared.destination,
                                    to: part,
                                    didAttach: &attachedProducer
                                ) {
                                    continuation.yield(identified)
                                }
                            }
                            if mapper.shouldTerminateTransport {
                                // Codex's response terminal can precede HTTP EOF.
                                // Stop the parser task so its URLSession byte
                                // iteration cancels the underlying request.
                                eventSource.cancel()
                                break
                            }
                        }

                        // Several protocols carry final usage nowhere else, so
                        // this runs even when the connection ended without a sentinel.
                        for part in mapper.finish() {
                            if case .streamStart = part { continue }
                            for identified in parts(
                                attaching: prepared.destination,
                                to: part,
                                didAttach: &attachedProducer
                            ) {
                                continuation.yield(identified)
                            }
                        }
                        if !attachedProducer {
                            continuation.yield(.responseMetadata(ResponseMetadata(
                                producer: prepared.destination
                            )))
                        }
                        break requestLoop
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Complete responses

    /// Sends a request and returns the complete response.
    ///
    /// The counterpart to ``stream(_:)`` for callers who want the outcome
    /// rather than the events — no loop, no delta assembly:
    ///
    /// ```swift
    /// let response = try await client.generate(options)
    /// response.text
    /// ```
    ///
    /// Providers with a supported complete-response transport receive a
    /// genuinely non-streaming request. OpenAI Codex is the exception: its
    /// supported transport is SSE, so this method collects ``stream(_:)`` into
    /// the same complete ``AIResponse``. To continue the conversation, append
    /// ``AIResponse/assistantMessage`` to the prompt.
    public func generate(_ options: CallOptions) async throws -> AIResponse {
        try await generate(prepare(options))
    }

    func generate(_ prepared: PreparedAIRequest) async throws -> AIResponse {
        let wire = prepared.wire
        let options = prepared.options
        if wire == .openAICodex {
            return try await stream(prepared).collect()
        }

        let encoded = encode(prepared, streaming: false)
        let request = try makeRequest(
            wire: wire, options: options, body: encoded.body, streaming: false
        )
        let (data, response) = try await session.data(for: request)

        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw AIClientError(kind: .http(
                status: http.statusCode,
                body: String(decoding: data, as: UTF8.self)
            ))
        }

        // Same contract as a stream: `streamStart` first, carrying the
        // encoding warnings, then the decoded parts.
        var parts: [StreamPart] = [.streamStart(warnings: encoded.warnings)]
        var attachedProducer = false
        for part in NonStreamingResponse.decode(try JSONValue.decode(from: data), wire: wire) {
            if case .streamStart = part { continue }
            parts.append(contentsOf: self.parts(
                attaching: prepared.destination,
                to: part,
                didAttach: &attachedProducer
            ))
        }
        if !attachedProducer {
            parts.append(.responseMetadata(ResponseMetadata(producer: prepared.destination)))
        }

        return AIResponse(parts: parts)
    }

    // MARK: - Request construction

    /// Resolves provider/API/model identity and model capabilities once for
    /// both transport paths.
    func prepare(_ options: CallOptions) throws -> PreparedAIRequest {
        let apiId = provider.adapter?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard let wire = WireProtocol(rawValue: apiId) else {
            throw AIClientError(kind: .unsupportedProtocol(apiId.isEmpty ? "none" : apiId))
        }

        let destination = ModelDestination(
            providerId: provider.id,
            apiId: apiId,
            modelId: options.model
        )
        return prepare(
            options,
            destination: destination,
            model: provider.model(options.model),
            wire: wire
        )
    }

    /// Materializes a validated selection without resolving identity or model
    /// capabilities a second time. The facade and direct-client paths meet at
    /// this boundary before either stream or complete transport begins.
    func prepare(
        _ options: CallOptions,
        destination: ModelDestination,
        model: ModelInfo?,
        wire: WireProtocol
    ) -> PreparedAIRequest {
        var preparedOptions = options
        preparedOptions.prompt = ConversationTransformer(
            destination: destination,
            model: model
        ).transform(options.prompt)

        // Request-level provider options follow the same destination isolation
        // rule as history metadata. Encoders still apply the trusted matching
        // namespace last, retaining the established override semantics.
        preparedOptions.providerOptions = options.providerOptions[destination.providerId]
            .map { [destination.providerId: $0] } ?? [:]

        return PreparedAIRequest(
            destination: destination,
            provider: provider,
            model: model,
            wire: wire,
            options: preparedOptions
        )
    }

    /// Adds requested identity to provider response metadata without adding a
    /// second event on normal responses. If a provider reports no metadata at
    /// all, a synthetic metadata part is emitted before `finish` when present,
    /// or before the stream closes otherwise.
    private func parts(
        attaching destination: ModelDestination,
        to part: StreamPart,
        didAttach: inout Bool
    ) -> [StreamPart] {
        if case .responseMetadata(var metadata) = part {
            metadata.producer = destination
            didAttach = true
            return [.responseMetadata(metadata)]
        }
        if case .finish = part, !didAttach {
            didAttach = true
            return [
                .responseMetadata(ResponseMetadata(producer: destination)),
                part,
            ]
        }
        return [part]
    }

    /// Encodes only from the immutable identity and capabilities resolved at
    /// the shared preparation boundary.
    func encode(
        _ prepared: PreparedAIRequest,
        streaming: Bool = true
    ) -> EncodedRequest {
        encode(
            prepared.options,
            provider: prepared.provider,
            wire: prepared.wire,
            model: prepared.model,
            streaming: streaming
        )
    }

    private func encode(
        _ options: CallOptions,
        provider: ProviderInfo,
        wire: WireProtocol,
        model: ModelInfo?,
        streaming: Bool
    ) -> EncodedRequest {
        switch wire {
        case .anthropicMessages:
            AnthropicMessagesRequest.encode(
                options,
                model: model,
                reasoningToggle: provider.reasoningToggle,
                providerId: provider.id,
                streaming: streaming
            )
        case .openAICompletions:
            // The dialect is selected from the provider, not the protocol:
            // thirty-eight providers share this encoder and disagree about the
            // details.
            OpenAICompletionsRequest.encode(
                options,
                model: model,
                dialect: .forProvider(provider.id),
                reasoningToggle: provider.reasoningToggle,
                providerId: provider.id,
                streaming: streaming
            )
        case .openAIResponses:
            OpenAIResponsesRequest.encode(
                options,
                model: model,
                providerId: provider.id,
                streaming: streaming
            )
        case .openAICodex:
            // Codex has no supported non-streaming transport. Its encoder is
            // intentionally independent of this shared dispatch flag.
            OpenAICodexResponsesRequest.encode(options, model: model, providerId: provider.id)
        case .googleGenerativeAI:
            // Gemini switches between streaming and not in the URL, not the body.
            GoogleGenerativeAIRequest.encode(options, model: model, providerId: provider.id)
        }
    }

    func makeRequest(
        wire: WireProtocol,
        options: CallOptions,
        body: JSONValue,
        streaming: Bool = true,
        oauthCredential: OAuthCredential? = nil
    ) throws -> URLRequest {
        let base = try resolveBaseURL(wire: wire)
        var request = URLRequest(
            url: endpoint(wire: wire, base: base, model: options.model, streaming: streaming)
        )

        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.setValue(
            streaming ? "text/event-stream" : "application/json",
            forHTTPHeaderField: "accept"
        )

        for (name, value) in authHeaders(wire: wire, oauthCredential: oauthCredential) {
            request.setValue(value, forHTTPHeaderField: name)
        }
        if wire == .openAICodex {
            let credential = oauthCredential ?? configuredOAuthCredential
            guard let credential,
                  !credential.accessToken.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            else { throw AIClientError(kind: .missingCodexCredential) }
            let accountId = OpenAICodexOAuthClient.accountId(from: credential.accessToken)
                ?? credential.accountId
            guard let accountId, !accountId.isEmpty else {
                throw AIClientError(kind: .missingCodexAccountId)
            }
            request.setValue(accountId, forHTTPHeaderField: "chatgpt-account-id")
            request.setValue("responses=experimental", forHTTPHeaderField: "OpenAI-Beta")
            request.setValue("aikit-swift", forHTTPHeaderField: "originator")
            request.setValue("AIKitSwift", forHTTPHeaderField: "User-Agent")
            if let sessionId = body["prompt_cache_key"]?.stringValue, !sessionId.isEmpty {
                request.setValue(sessionId, forHTTPHeaderField: "session-id")
                request.setValue(sessionId, forHTTPHeaderField: "x-client-request-id")
            }
        }
        for (name, value) in configuration.extraHeaders {
            request.setValue(value, forHTTPHeaderField: name)
        }

        // Sorted keys: prompt caching is a byte-level prefix match, so an
        // unstable key order silently destroys every cache hit.
        request.httpBody = Data((try? body.encodedString())?.utf8 ?? "{}".utf8)

        return request
    }

    func resolveBaseURL() throws -> URL {
        if let baseURL = configuration.baseURL { return baseURL }
        guard let api = provider.api, let url = URL(string: api) else {
            throw AIClientError(kind: .missingBaseURL(provider.id))
        }
        return url
    }

    private func resolveBaseURL(wire: WireProtocol) throws -> URL {
        if let baseURL = configuration.baseURL { return baseURL }
        if let api = provider.api, let url = URL(string: api) { return url }
        if wire == .openAICodex {
            return URL(string: "https://chatgpt.com/backend-api")!
        }
        throw AIClientError(kind: .missingBaseURL(provider.id))
    }

    func endpoint(wire: WireProtocol, base: URL, model: String, streaming: Bool = true) -> URL {
        // Catalog base URLs are inconsistent about including a version prefix,
        // so it is added only when absent.
        func path(_ versioned: String, _ bare: String) -> String {
            base.path.hasSuffix("/\(versioned)") ? bare : "\(versioned)/\(bare)"
        }

        switch wire {
        case .anthropicMessages:
            return base.appending(path: path("v1", "messages"))
        case .openAICompletions:
            return base.appending(path: path("v1", "chat/completions"))
        case .openAIResponses:
            return base.appending(path: path("v1", "responses"))
        case .openAICodex:
            let normalized = base.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            if normalized.hasSuffix("/codex/responses") {
                return URL(string: normalized) ?? base
            }
            if normalized.hasSuffix("/codex") {
                return URL(string: "\(normalized)/responses") ?? base
            }
            return URL(string: "\(normalized)/codex/responses") ?? base
        case .googleGenerativeAI:
            // Gemini switches methods rather than taking a `stream` flag, and
            // streaming needs an explicit opt-in to Server-Sent Events —
            // without `alt=sse` it streams a JSON array.
            guard streaming else {
                return base.appending(path: path("v1beta", "models/\(model):generateContent"))
            }
            let url = base.appending(path: path("v1beta", "models/\(model):streamGenerateContent"))
            return URL(string: "\(url.absoluteString)?alt=sse") ?? url
        }
    }

    func authHeaders(wire: WireProtocol) -> [String: String] {
        authHeaders(wire: wire, oauthCredential: nil)
    }

    private func authHeaders(
        wire: WireProtocol,
        oauthCredential: OAuthCredential?
    ) -> [String: String] {
        if let oauthCredential {
            return ["authorization": "Bearer \(oauthCredential.accessToken)"]
        }
        switch configuration.authorization {
        case .none:
            // Local runtimes and gateways often need none.
            return wire == .anthropicMessages ? ["anthropic-version": "2023-06-01"] : [:]

        case .apiKey(let key):
            switch wire {
            case .anthropicMessages:
                return [
                    "x-api-key": key,
                    // Required on every request; omitting it is a 400.
                    "anthropic-version": "2023-06-01",
                ]
            case .openAICompletions, .openAIResponses, .openAICodex:
                return ["authorization": "Bearer \(key)"]
            case .googleGenerativeAI:
                return ["x-goog-api-key": key]
            }

        case .oauth(let credential):
            // An OAuth token is a bearer token on `Authorization` — *not* an
            // API key on `x-api-key`. Converting a working request from a key
            // to a token is a header change, not a value swap, and sending it
            // the wrong way fails as an authentication error that looks like a
            // bad token.
            var headers = ["authorization": "Bearer \(credential.accessToken)"]

            if wire == .anthropicMessages {
                headers["anthropic-version"] = "2023-06-01"
                // Anthropic gates OAuth-authenticated requests behind this
                // opt-in; without it `/v1/messages` rejects the token.
                headers["anthropic-beta"] = "oauth-2025-04-20"
            }

            return headers
        }
    }

    private var configuredOAuthCredential: OAuthCredential? {
        switch configuration.authorization {
        case .oauth(let credential): credential
        case .apiKey(let token): OAuthCredential(accessToken: token)
        case .none: nil
        }
    }

    private func credentialForRequest(wire: WireProtocol) async throws -> OAuthCredential? {
        if wire == .openAICodex, let provider = configuration.oauthCredentialProvider {
            return try await provider.credential()
        }
        return configuredOAuthCredential
    }
}
