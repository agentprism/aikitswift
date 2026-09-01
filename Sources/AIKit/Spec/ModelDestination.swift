import Foundation

/// The complete identity of the model selected for one request.
///
/// Provider and model ids are not globally unique, and changing the API changes
/// how history may be replayed. All three values therefore participate in
/// equality and hashing.
public struct ModelDestination: Sendable, Hashable, Codable, CustomStringConvertible {
    /// The provider that owns configuration, credentials, and the endpoint.
    public var providerId: String
    /// The provider's declared adapter/API protocol id.
    public var apiId: String
    /// The model id as that provider expects it on the wire.
    public var modelId: String

    public init(providerId: String, apiId: String, modelId: String) {
        self.providerId = providerId
        self.apiId = apiId
        self.modelId = modelId
    }

    public var description: String {
        "\(providerId)/\(apiId)/\(modelId)"
    }
}
