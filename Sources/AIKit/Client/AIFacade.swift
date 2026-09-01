import Foundation

#if canImport(FoundationNetworking)
import FoundationNetworking
#endif

/// A destination or provider configuration could not be resolved by
/// ``AIFacade``.
public struct AIFacadeError: Error, Sendable, CustomStringConvertible {
    public enum Kind: Sendable, Hashable {
        case unknownProvider(String)
        case missingModel(providerId: String, modelId: String)
        case missingConfiguration(String)
        case missingAPI(String)
        case apiMismatch(providerId: String, requested: String, declared: String)
        case unsupportedProtocol(String)
    }

    public var kind: Kind

    public init(kind: Kind) {
        self.kind = kind
    }

    public var description: String {
        switch kind {
        case .unknownProvider(let id):
            "Unknown provider '\(id)'. Register it with the facade or use a catalog provider."
        case .missingModel(let providerId, let modelId):
            "Provider '\(providerId)' does not declare model '\(modelId)'."
        case .missingConfiguration(let providerId):
            "Provider '\(providerId)' has no facade configuration. Supply an AIClient.Configuration for this provider, even when it intentionally uses no credential."
        case .missingAPI(let providerId):
            "Provider '\(providerId)' does not declare an adapter/API protocol."
        case .apiMismatch(let providerId, let requested, let declared):
            "Destination API '\(requested)' does not match provider '\(providerId)' declaration '\(declared)'."
        case .unsupportedProtocol(let apiId):
            "Provider API '\(apiId)' is not implemented by AIKit."
        }
    }
}

extension AIFacadeError: LocalizedError {
    public var errorDescription: String? { description }
}

/// One provider-neutral entry point for streaming and complete model calls.
///
/// A facade owns an immutable provider registry, provider-scoped client
/// configurations, and one injected ``URLSession``. The destination changes on
/// each ``stream(_:)`` or ``generate(_:)`` call; endpoint resolution,
/// authentication, request encoding, SSE parsing, retries, and response mapping
/// continue through the existing ``AIClient`` implementation.
public struct AIFacade: Sendable {
    private let providers: [String: ProviderInfo]
    private let configurations: [String: AIClient.Configuration]
    private let session: URLSession

    /// Creates a facade over the supplied provider registry.
    ///
    /// Pass `ProviderCatalog.all` (the default) for bundled providers, or a
    /// registry containing user-created ``ProviderInfo`` values for custom
    /// endpoints. Configurations are keyed by provider id and are never shared
    /// or merged across providers.
    public init(
        providers: [ProviderInfo] = ProviderCatalog.all,
        configurations: [String: AIClient.Configuration],
        session: URLSession = .shared
    ) {
        self.providers = Dictionary(
            providers.map { ($0.id, $0) },
            uniquingKeysWith: { _, latest in latest }
        )
        self.configurations = configurations
        self.session = session
    }

    /// Selects a catalog or registered custom model and derives its API from
    /// the provider's declared adapter.
    public func destination(providerId: String, modelId: String) throws -> ModelDestination {
        let provider = try requireProvider(providerId)
        let apiId = try requireAPI(provider)
        _ = try requireSupportedWire(apiId)
        _ = try requireModel(modelId, from: provider)
        return ModelDestination(providerId: providerId, apiId: apiId, modelId: modelId)
    }

    /// Returns the selected model's normalized capabilities after validating
    /// the entire destination identity.
    public func model(for destination: ModelDestination) throws -> ModelInfo {
        try resolve(destination).model
    }

    /// Streams normalized ``StreamPart`` values from the selected destination.
    public func stream(_ request: AIRequest) throws -> AsyncThrowingStream<StreamPart, any Error> {
        let (client, prepared) = try prepare(request)
        return try client.stream(prepared)
    }

    /// Returns one normalized ``AIResponse`` from the selected destination.
    public func generate(_ request: AIRequest) async throws -> AIResponse {
        let (client, prepared) = try prepare(request)
        return try await client.generate(prepared)
    }

    private func prepare(_ request: AIRequest) throws -> (AIClient, PreparedAIRequest) {
        let selection = try resolve(request.destination)
        guard let configuration = configurations[selection.provider.id] else {
            throw AIFacadeError(kind: .missingConfiguration(selection.provider.id))
        }

        let client = AIClient(
            provider: selection.provider,
            configuration: configuration,
            session: session
        )
        // Destination validation above provides facade-specific diagnostics;
        // materialize that exact identity and capability selection once at the
        // AIClient boundary shared by stream and complete transport.
        let prepared = client.prepare(
            request.callOptions,
            destination: request.destination,
            model: selection.model,
            wire: selection.wire
        )
        return (client, prepared)
    }

    private func resolve(
        _ destination: ModelDestination
    ) throws -> (provider: ProviderInfo, model: ModelInfo, wire: WireProtocol) {
        let provider = try requireProvider(destination.providerId)
        let apiId = try requireAPI(provider)
        guard destination.apiId == apiId else {
            throw AIFacadeError(kind: .apiMismatch(
                providerId: provider.id,
                requested: destination.apiId,
                declared: apiId
            ))
        }
        let wire = try requireSupportedWire(apiId)
        let model = try requireModel(destination.modelId, from: provider)
        return (provider, model, wire)
    }

    private func requireProvider(_ id: String) throws -> ProviderInfo {
        guard let provider = providers[id] else {
            throw AIFacadeError(kind: .unknownProvider(id))
        }
        return provider
    }

    private func requireAPI(_ provider: ProviderInfo) throws -> String {
        let apiId = provider.adapter?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        guard !apiId.isEmpty else {
            throw AIFacadeError(kind: .missingAPI(provider.id))
        }
        return apiId
    }

    private func requireSupportedWire(_ apiId: String) throws -> WireProtocol {
        guard let wire = WireProtocol(rawValue: apiId) else {
            throw AIFacadeError(kind: .unsupportedProtocol(apiId))
        }
        return wire
    }

    private func requireModel(_ id: String, from provider: ProviderInfo) throws -> ModelInfo {
        guard let model = provider.model(id) else {
            throw AIFacadeError(kind: .missingModel(providerId: provider.id, modelId: id))
        }
        return model
    }
}
