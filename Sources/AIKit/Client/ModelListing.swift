import Foundation

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// Asking an endpoint what it can serve.
///
/// The catalog cannot answer this for everyone: Ollama and LM Studio serve
/// whatever the user pulled locally, and a self-hosted gateway serves whatever
/// its operator wired up. That is the "Fetch models" button in a settings
/// panel, and it belongs next to the protocols rather than in each app — the
/// endpoint is `GET /models` in three mutually incompatible response shapes.
extension AIClient {

    /// Models this endpoint reports, enriched from the catalog where it knows
    /// them.
    ///
    /// The live list is authoritative about *existence*; the catalog is
    /// authoritative about capability, since no `/models` response carries
    /// context windows or pricing. A model the catalog knows comes back with
    /// its full metadata, and anything else comes back as a bare id.
    public func models() async throws -> [ModelInfo] {
        guard let wire = provider.wireProtocol else {
            throw AIClientError(kind: .unsupportedProtocol(provider.adapter ?? "none"))
        }

        let base = try resolveBaseURL()
        var request = URLRequest(url: Self.modelsEndpoint(wire: wire, base: base))
        request.httpMethod = "GET"
        request.setValue("application/json", forHTTPHeaderField: "accept")

        for (name, value) in authHeaders(wire: wire) {
            request.setValue(value, forHTTPHeaderField: name)
        }
        for (name, value) in configuration.extraHeaders {
            request.setValue(value, forHTTPHeaderField: name)
        }

        let (data, response) = try await session.data(for: request)

        if let http = response as? HTTPURLResponse, !(200..<300).contains(http.statusCode) {
            throw AIClientError(kind: .http(
                status: http.statusCode,
                body: String(decoding: data, as: UTF8.self)
            ))
        }

        let listed = Self.decodeModels(try JSONValue.decode(from: data), wire: wire)

        return listed.map { model in
            // Prefer this provider's own catalog entry, then any provider's:
            // resellers list the same model and the capabilities travel with
            // the model, not the vendor.
            provider.model(model.id)
                ?? ProviderCatalog.model(model.id)?.1
                ?? model
        }
    }

    /// Lists models from an endpoint without a catalog entry to stand behind
    /// it — a local runtime, or a gateway the user just typed a URL for.
    public static func models(
        at endpoint: URL,
        speaking wire: WireProtocol = .openAICompletions,
        authorization: Authorization = .none,
        extraHeaders: [String: String] = [:],
        session: URLSession = .shared
    ) async throws -> [ModelInfo] {
        let provider = ProviderInfo(id: "custom", api: endpoint.absoluteString, speaking: wire)
        let client = AIClient(
            provider: provider,
            configuration: .init(authorization: authorization, baseURL: endpoint, extraHeaders: extraHeaders),
            session: session
        )
        return try await client.models()
    }

    // MARK: - Endpoints

    static func modelsEndpoint(wire: WireProtocol, base: URL) -> URL {
        func path(_ versioned: String, _ bare: String) -> String {
            base.path.hasSuffix("/\(versioned)") ? bare : "\(versioned)/\(bare)"
        }

        switch wire {
        case .anthropicMessages:
            return base.appending(path: path("v1", "models"))
        case .openAICompletions, .openAIResponses, .openAICodex:
            return base.appending(path: path("v1", "models"))
        case .googleGenerativeAI:
            return base.appending(path: path("v1beta", "models"))
        }
    }

    // MARK: - Response shapes

    static func decodeModels(_ body: JSONValue, wire: WireProtocol) -> [ModelInfo] {
        switch wire {
        case .googleGenerativeAI:
            // `models: [{ name: "models/gemini-3-flash", displayName, ... }]`,
            // and the list includes embedding and TTS models that cannot serve
            // a chat request at all.
            let models = body["models"]?.arrayValue ?? []
            return models.compactMap { entry in
                guard let name = entry["name"]?.stringValue else { return nil }
                let methods = entry["supportedGenerationMethods"]?.arrayValue?
                    .compactMap(\.stringValue) ?? []
                guard methods.isEmpty || methods.contains("generateContent") else { return nil }

                return ModelInfo(
                    // The id is path-qualified on the wire but not in a request.
                    id: name.hasPrefix("models/") ? String(name.dropFirst("models/".count)) : name,
                    name: entry["displayName"]?.stringValue,
                    description: entry["description"]?.stringValue,
                    limit: ModelInfo.Limit(
                        context: entry["inputTokenLimit"]?.intValue,
                        output: entry["outputTokenLimit"]?.intValue
                    )
                )
            }

        case .anthropicMessages, .openAICompletions, .openAIResponses, .openAICodex:
            // `data: [{ id, ... }]` — Anthropic adds `display_name`, and the
            // OpenAI-compatible crowd carries nothing beyond the id, which is
            // exactly why the catalog is consulted afterwards.
            let models = body["data"]?.arrayValue ?? []
            return models.compactMap { entry in
                guard let id = entry["id"]?.stringValue else { return nil }
                return ModelInfo(id: id, name: entry["display_name"]?.stringValue)
            }
        }
    }
}
