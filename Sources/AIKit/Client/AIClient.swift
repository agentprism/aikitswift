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
        }
    }
}

// Without this, `error.localizedDescription` — the accessor every app reaches
// for first — shows Foundation's generic text instead of the message above.
extension AIClientError: LocalizedError {
    public var errorDescription: String? { description }
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

        public init(
            authorization: Authorization = .none,
            baseURL: URL? = nil,
            extraHeaders: [String: String] = [:]
        ) {
            self.authorization = authorization
            self.baseURL = baseURL
            self.extraHeaders = extraHeaders
        }

        public init(apiKey: String?, baseURL: URL? = nil, extraHeaders: [String: String] = [:]) {
            self.init(
                authorization: apiKey.map(Authorization.apiKey) ?? .none,
                baseURL: baseURL,
                extraHeaders: extraHeaders
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
        guard let wire = provider.wireProtocol else {
            throw AIClientError(kind: .unsupportedProtocol(provider.adapter ?? "none"))
        }

        let model = provider.model(options.model)
        let encoded = encode(options, wire: wire, model: model)
        let request = try makeRequest(wire: wire, options: options, body: encoded.body)

        let session = self.session

        return AsyncThrowingStream { continuation in
            let task = Task {
                do {
                    // Warnings come from encoding, so the client emits
                    // `stream-start` itself and suppresses the mapper's. The
                    // contract stays: exactly one, first, carrying warnings.
                    continuation.yield(.streamStart(warnings: encoded.warnings))

                    let (bytes, response) = try await session.bytes(for: request)

                    if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
                        // The body holds the provider's error detail, which is
                        // the only useful thing about a failed request — so it
                        // is collected verbatim. Reading it as lines would drop
                        // the line breaks and run the words together.
                        var detail: [UInt8] = []
                        for try await byte in bytes { detail.append(byte) }
                        throw AIClientError(kind: .http(
                            status: http.statusCode,
                            body: String(decoding: detail, as: UTF8.self)
                        ))
                    }

                    var mapper = wire.makeMapper()

                    for try await event in sseEvents(from: bytes) {
                        for part in mapper.map(rawJSON: event.data) {
                            if case .streamStart = part { continue }
                            continuation.yield(part)
                        }
                    }

                    // Several protocols carry final usage nowhere else, so this
                    // runs even when the connection ended without a sentinel.
                    for part in mapper.finish() {
                        if case .streamStart = part { continue }
                        continuation.yield(part)
                    }

                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
            continuation.onTermination = { _ in task.cancel() }
        }
    }

    // MARK: - Non-streaming

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
    /// This is a genuinely non-streaming request on the wire, not a drained
    /// stream. The body is decoded through the same mappers as a stream would
    /// be, so the two paths cannot drift. To continue the conversation, append
    /// ``AIResponse/assistantMessage`` to the prompt.
    public func generate(_ options: CallOptions) async throws -> AIResponse {
        guard let wire = provider.wireProtocol else {
            throw AIClientError(kind: .unsupportedProtocol(provider.adapter ?? "none"))
        }

        let model = provider.model(options.model)
        let encoded = encode(options, wire: wire, model: model, streaming: false)
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
        for part in NonStreamingResponse.decode(try JSONValue.decode(from: data), wire: wire) {
            if case .streamStart = part { continue }
            parts.append(part)
        }

        return AIResponse(parts: parts)
    }

    // MARK: - Request construction

    func encode(
        _ options: CallOptions,
        wire: WireProtocol,
        model: ModelInfo?,
        streaming: Bool = true
    ) -> EncodedRequest {
        switch wire {
        case .anthropicMessages:
            AnthropicMessagesRequest.encode(
                options,
                model: model,
                reasoningToggle: provider.reasoningToggle,
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
                streaming: streaming
            )
        case .openAIResponses, .openAICodex:
            OpenAIResponsesRequest.encode(options, model: model, streaming: streaming)
        case .googleGenerativeAI:
            // Gemini switches between streaming and not in the URL, not the body.
            GoogleGenerativeAIRequest.encode(options, model: model)
        }
    }

    func makeRequest(
        wire: WireProtocol,
        options: CallOptions,
        body: JSONValue,
        streaming: Bool = true
    ) throws -> URLRequest {
        let base = try resolveBaseURL()
        var request = URLRequest(
            url: endpoint(wire: wire, base: base, model: options.model, streaming: streaming)
        )

        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "content-type")
        request.setValue(
            streaming ? "text/event-stream" : "application/json",
            forHTTPHeaderField: "accept"
        )

        for (name, value) in authHeaders(wire: wire) {
            request.setValue(value, forHTTPHeaderField: name)
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
        case .openAIResponses, .openAICodex:
            return base.appending(path: path("v1", "responses"))
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
}
