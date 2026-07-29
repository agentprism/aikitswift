import Foundation

/// What a model can do, and what it costs to talk to.
///
/// Sourced from the vendored catalog rather than hardcoded, so capability
/// changes track upstream instead of rotting in Swift.
public struct ModelInfo: Sendable, Hashable, Codable {
    public struct Reasoning: Sendable, Hashable, Codable {
        public var supported: Bool?
        /// Whether reasoning is on when the request says nothing.
        ///
        /// Worth checking before sizing `maxOutputTokens`: where reasoning
        /// defaults on, the budget covers thinking *and* the answer.
        public var `default`: Bool?
    }

    public struct Limit: Sendable, Hashable, Codable {
        /// Context window, in tokens. The denominator for a context-usage report.
        public var context: Int?
        /// Maximum output tokens per request.
        public var output: Int?
    }

    public struct Modalities: Sendable, Hashable, Codable {
        public var input: [String]?
        public var output: [String]?
    }

    public var id: String
    public var name: String?
    public var description: String?
    public var family: String?

    /// Whether files can be attached to a prompt.
    public var attachment: Bool?
    public var reasoning: Reasoning?
    public var toolCall: Bool?
    public var structuredOutput: Bool?

    /// Whether the model accepts a sampling temperature at all.
    ///
    /// `false` on newer Anthropic models, which reject the parameter outright
    /// rather than ignoring it. Encoders consult this to drop the setting and
    /// warn instead of letting the request fail with a 400.
    public var temperature: Bool?

    public var limit: Limit?
    public var modalities: Modalities?
    public var openWeights: Bool?
    public var knowledge: String?
    public var releaseDate: String?
    public var lastUpdated: String?

    public var contextWindow: Int? { limit?.context }
    public var maxOutputTokens: Int? { limit?.output }
    public var supportsTools: Bool { toolCall ?? false }
    public var supportsReasoning: Bool { reasoning?.supported ?? false }
    /// Newer models reject `temperature`; absence of the flag means unknown, so
    /// it is treated as accepted.
    public var supportsTemperature: Bool { temperature ?? true }
    public var supportsVision: Bool { modalities?.input?.contains("image") ?? false }
}

/// A provider: where to send a request, how to authenticate, and — the field
/// that matters most — which wire protocol it speaks.
public struct ProviderInfo: Sendable, Hashable, Codable {
    public var id: String
    public var name: String?
    /// Base URL. Absent for providers whose endpoint is configured per install
    /// (self-hosted gateways, local runtimes).
    public var api: String?
    /// The wire protocol, as named upstream.
    ///
    /// Kept as a raw string so an adapter this library has not implemented yet
    /// still loads instead of failing the whole catalog.
    public var adapter: String?
    public var models: [ModelInfo]?

    /// The implemented protocol for this provider, if there is one.
    public var wireProtocol: WireProtocol? {
        adapter.flatMap(WireProtocol.init(rawValue:))
    }

    public func model(_ id: String) -> ModelInfo? {
        models?.first { $0.id == id }
    }
}

/// The bundled provider catalog.
///
/// Loaded once, lazily, from JSON shipped with the package. Adding a provider
/// upstream needs no Swift changes here at all.
public enum ProviderCatalog {

    /// Every provider in the catalog, sorted by id.
    public static let all: [ProviderInfo] = load()

    private static let byId: [String: ProviderInfo] = Dictionary(
        all.map { ($0.id, $0) },
        uniquingKeysWith: { first, _ in first }
    )

    public static func provider(_ id: String) -> ProviderInfo? { byId[id] }

    /// Providers speaking a given protocol.
    public static func providers(speaking wire: WireProtocol) -> [ProviderInfo] {
        all.filter { $0.wireProtocol == wire }
    }

    /// Finds a model by id, searching every provider.
    ///
    /// Model ids are not globally unique — several providers resell the same
    /// model — so this returns the first match. Pass a provider id when the
    /// distinction matters.
    public static func model(_ modelId: String, provider providerId: String? = nil) -> (ProviderInfo, ModelInfo)? {
        let candidates = providerId.flatMap { byId[$0].map { [$0] } } ?? all

        for provider in candidates {
            if let model = provider.model(modelId) { return (provider, model) }
        }
        return nil
    }

    private static func load() -> [ProviderInfo] {
        guard let directory = Bundle.module.resourceURL?.appending(path: "Catalog/providers"),
              let files = try? FileManager.default.contentsOfDirectory(
                  at: directory, includingPropertiesForKeys: nil
              )
        else {
            return []
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        return files
            .filter { $0.pathExtension == "json" }
            .compactMap { url in
                guard let data = try? Data(contentsOf: url) else { return nil }
                // A provider whose schema drifted is skipped rather than
                // taking the whole catalog down with it.
                return try? decoder.decode(ProviderInfo.self, from: data)
            }
            .sorted { $0.id < $1.id }
    }
}
