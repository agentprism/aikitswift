import Foundation

/// Prepares normalized conversation history for one destination model.
///
/// The transformer is a deterministic value operation: it does not mutate the
/// caller's prompt, perform I/O, or depend on provider wire payload types.
/// Opaque state is retained only when an assistant message's complete producer
/// identity exactly equals ``destination``.
public struct ConversationTransformer: Sendable {
    /// Replacement for one or more adjacent unsupported user images.
    public static let omittedUserImage = "(image omitted: model does not support images)"
    /// Replacement for one or more adjacent unsupported tool-result images.
    public static let omittedToolImage = "(tool image omitted: model does not support images)"
    /// Content of a synthetic error result for an unresolved client tool call.
    public static let missingToolResult = "No result provided"

    /// Complete destination identity used for exact replay decisions.
    public let destination: ModelDestination
    /// Resolved capabilities for `destination`, or `nil` when unknown.
    public let model: ModelInfo?

    /// Creates a transformer for a resolved destination and its capabilities.
    ///
    /// A `nil` model means capabilities are unknown, so images are retained.
    public init(destination: ModelDestination, model: ModelInfo?) {
        self.destination = destination
        self.model = model
    }

    /// Returns destination-valid history without changing `prompt`.
    public func transform(_ prompt: Prompt) -> Prompt {
        var result: Prompt = []
        var associations: [ToolAssociation] = []
        var usedDestinationIds: Set<String> = []
        var callSequence = 0

        func insertMissingResults() {
            for association in associations
            where !association.providerExecuted && !association.hasResult {
                result.append(Message(role: .tool, content: [.toolResult(ToolResult(
                    toolCallId: association.destinationId,
                    toolName: association.toolName,
                    result: .string(Self.missingToolResult),
                    content: [.text(Self.missingToolResult)],
                    isError: true
                ))]))
            }
            associations.removeAll(keepingCapacity: true)
        }

        for message in prompt {
            switch message.role {
            case .assistant:
                insertMissingResults()
                guard message.outcome == nil else { continue }

                let exactMatch = message.producer == destination
                var transformed = message
                transformed.providerOptions = exactMatch ? message.providerOptions : nil
                transformed.content = message.content.compactMap { part in
                    switch part {
                    case .reasoning(let text, _) where !exactMatch:
                        return text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            ? nil
                            : .text(text)

                    case .toolCall(var call):
                        callSequence += 1
                        let sourceId = call.toolCallId
                        let sourceItemId = Self.openAIItemId(in: call.providerMetadata)

                        if exactMatch {
                            usedDestinationIds.insert(call.toolCallId)
                        } else {
                            let normalized = normalizedToolCall(
                                call,
                                sourceItemId: sourceItemId,
                                sequence: callSequence,
                                used: &usedDestinationIds
                            )
                            call.toolCallId = normalized.callId
                            call.namespace = nil
                            call.providerMetadata = normalized.providerMetadata
                        }

                        associations.append(ToolAssociation(
                            sourceId: sourceId,
                            sourceItemId: sourceItemId,
                            destinationId: call.toolCallId,
                            toolName: call.toolName,
                            providerExecuted: call.providerExecuted,
                            exactMatch: exactMatch
                        ))
                        return .toolCall(call)

                    case .toolResult(let toolResult):
                        return .toolResult(transformToolResult(
                            toolResult,
                            fallbackExactMatch: exactMatch,
                            associations: &associations
                        ))

                    default:
                        return part
                    }
                }
                result.append(transformed)

            case .tool:
                var transformed = message
                transformed.providerOptions = optionsForDestination(message.providerOptions)
                transformed.content = message.content.map { part in
                    guard case .toolResult(let toolResult) = part else { return part }
                    return .toolResult(transformToolResult(
                        toolResult,
                        fallbackExactMatch: false,
                        associations: &associations
                    ))
                }
                result.append(transformed)

            case .user:
                insertMissingResults()
                var transformed = message
                transformed.providerOptions = optionsForDestination(message.providerOptions)
                if model?.supportsVision == false {
                    transformed.content = replaceUserImages(in: message.content)
                }
                result.append(transformed)

            case .system:
                var transformed = message
                transformed.providerOptions = optionsForDestination(message.providerOptions)
                result.append(transformed)
            }
        }

        insertMissingResults()
        return result
    }

    private func transformToolResult(
        _ source: ToolResult,
        fallbackExactMatch: Bool,
        associations: inout [ToolAssociation]
    ) -> ToolResult {
        var transformed = source
        let sourceItemId = Self.openAIItemId(in: source.providerMetadata)
        let associationIndex = associationIndex(
            for: source,
            sourceItemId: sourceItemId,
            in: associations
        )
        let exactMatch: Bool

        if let associationIndex {
            transformed.toolCallId = associations[associationIndex].destinationId
            associations[associationIndex].hasResult = true
            exactMatch = associations[associationIndex].exactMatch
        } else {
            exactMatch = fallbackExactMatch
        }

        if !exactMatch {
            transformed.providerMetadata = nil
        }
        if model?.supportsVision == false, let content = source.content {
            transformed.content = replaceToolImages(in: content)
        }
        return transformed
    }

    private func associationIndex(
        for result: ToolResult,
        sourceItemId: String?,
        in associations: [ToolAssociation]
    ) -> Int? {
        if let sourceItemId,
           let exact = associations.firstIndex(where: {
               $0.sourceId == result.toolCallId && $0.sourceItemId == sourceItemId
           }) {
            return exact
        }
        if let destination = associations.firstIndex(where: {
            $0.destinationId == result.toolCallId && !$0.hasResult
        }) {
            return destination
        }
        if let source = associations.firstIndex(where: {
            $0.sourceId == result.toolCallId && !$0.hasResult
        }) {
            return source
        }
        return associations.firstIndex {
            $0.destinationId == result.toolCallId || $0.sourceId == result.toolCallId
        }
    }

    private func normalizedToolCall(
        _ call: ToolCall,
        sourceItemId: String?,
        sequence: Int,
        used: inout Set<String>
    ) -> (callId: String, providerMetadata: ProviderMetadata?) {
        let seed = sourceItemId.map { "\(call.toolCallId)|\($0)" }
            ?? "\(call.toolCallId)|\(sequence)"

        switch WireProtocol(rawValue: destination.apiId) {
        case .anthropicMessages:
            let candidate = Self.sanitize(call.toolCallId, maximumLength: 64)
            return (Self.unique(candidate, seed: seed, limit: 64, used: &used), nil)

        case .openAICompletions:
            let candidate = Self.openAICompletionsId(
                callId: call.toolCallId,
                itemId: sourceItemId,
                normalizeOrdinaryId: destination.providerId == "openai"
            )
            return (Self.unique(candidate, seed: seed, limit: 40, used: &used), nil)

        case .openAIResponses, .openAICodex:
            let candidate = Self.responsesIdPart(call.toolCallId)
            let callId = Self.unique(candidate, seed: seed, limit: 64, used: &used)
            let supportsItemIds = ["openai", "openai-codex", "opencode"]
                .contains(destination.providerId)
            guard supportsItemIds else { return (callId, nil) }
            let alreadyTransformed = call.providerMetadata?["openai"]?["transformedItemId"]?.boolValue
                == true
            let itemId = sourceItemId.map {
                alreadyTransformed ? $0 : "fc_\(Self.shortHash($0))"
            }
            let metadata: ProviderMetadata? = itemId.map {
                ["openai": [
                    "itemId": JSONValue.string($0),
                    "transformedItemId": true,
                ]]
            }
            return (callId, metadata)

        case .googleGenerativeAI:
            guard Self.googleRequiresToolCallId(destination.modelId) else {
                used.insert(call.toolCallId)
                return (call.toolCallId, nil)
            }
            let candidate = Self.sanitize(call.toolCallId, maximumLength: 64)
            return (Self.unique(candidate, seed: seed, limit: 64, used: &used), nil)

        case nil:
            used.insert(call.toolCallId)
            return (call.toolCallId, nil)
        }
    }

    private func optionsForDestination(_ options: ProviderMetadata?) -> ProviderMetadata? {
        guard let values = options?[destination.providerId] else { return nil }
        return [destination.providerId: values]
    }

    private func replaceUserImages(in content: [ContentPart]) -> [ContentPart] {
        var result: [ContentPart] = []
        var previousWasPlaceholder = false

        for part in content {
            if case .file(let file) = part, Self.isImage(file) {
                if !previousWasPlaceholder {
                    result.append(.text(Self.omittedUserImage))
                }
                previousWasPlaceholder = true
            } else {
                result.append(part)
                if case .text(let text) = part {
                    previousWasPlaceholder = text == Self.omittedUserImage
                } else {
                    previousWasPlaceholder = false
                }
            }
        }
        return result
    }

    private func replaceToolImages(in content: [ToolResultContent]) -> [ToolResultContent] {
        var result: [ToolResultContent] = []
        var previousWasPlaceholder = false

        for part in content {
            if case .file(let file) = part, Self.isImage(file) {
                if !previousWasPlaceholder {
                    result.append(.text(Self.omittedToolImage))
                }
                previousWasPlaceholder = true
            } else {
                result.append(part)
                if case .text(let text) = part {
                    previousWasPlaceholder = text == Self.omittedToolImage
                } else {
                    previousWasPlaceholder = false
                }
            }
        }
        return result
    }

    private static func isImage(_ file: FilePart) -> Bool {
        file.mediaType.lowercased().hasPrefix("image/")
    }

    private static func openAIItemId(in metadata: ProviderMetadata?) -> String? {
        metadata?["openai"]?["item"]?["id"]?.stringValue
            ?? metadata?["openai"]?["itemId"]?.stringValue
    }

    private static func sanitize(_ id: String, maximumLength: Int) -> String {
        let sanitized = id.unicodeScalars.map { scalar -> Character in
            let value = scalar.value
            let valid = (65...90).contains(value) || (97...122).contains(value)
                || (48...57).contains(value) || value == 45 || value == 95
            return valid ? Character(String(scalar)) : "_"
        }
        return String(sanitized.prefix(maximumLength))
    }

    private static func responsesIdPart(_ id: String) -> String {
        let value = sanitize(id, maximumLength: 64)
            .replacing(/_+$/, with: "")
        return value.isEmpty ? "call_\(shortHash(id))" : value
    }

    private static func openAICompletionsId(
        callId: String,
        itemId: String?,
        normalizeOrdinaryId: Bool
    ) -> String {
        let combined = itemId.map { "\(callId)|\($0)" } ?? callId
        guard let separator = combined.firstIndex(of: "|") else {
            return normalizeOrdinaryId ? String(callId.prefix(40)) : callId
        }

        let left = sanitize(String(combined[..<separator]), maximumLength: Int.max)
        let right = sanitize(String(combined[combined.index(after: separator)...]), maximumLength: Int.max)
        let candidate = right.isEmpty ? left : "\(left)_\(right)"
        guard candidate.count > 40 else { return candidate }
        let hash = String(shortHash(combined).prefix(8))
        let prefixLength = Swift.max(1, 40 - hash.count - 1)
        return "\(left.prefix(prefixLength))_\(hash)"
    }

    private static func unique(
        _ candidate: String,
        seed: String,
        limit: Int,
        used: inout Set<String>
    ) -> String {
        let candidate = candidate.isEmpty ? "call_\(shortHash(seed))" : candidate
        if used.insert(candidate).inserted { return candidate }

        var attempt = 0
        while true {
            let hashSeed = attempt == 0 ? seed : "\(seed)|\(attempt)"
            let suffix = "_\(shortHash(hashSeed).prefix(8))"
            let prefix = candidate.prefix(Swift.max(1, limit - suffix.count))
            let alternative = "\(prefix)\(suffix)"
            if used.insert(alternative).inserted { return alternative }
            attempt += 1
        }
    }

    static func googleRequiresToolCallId(_ modelId: String) -> Bool {
        let lowercased = modelId.lowercased()
        if lowercased.hasPrefix("claude-") || lowercased.hasPrefix("gpt-oss-") {
            return true
        }
        let pattern = /^gemini(?:-live)?-(\d+)/
        guard let match = lowercased.firstMatch(of: pattern),
              let major = Int(match.1) else { return false }
        return major >= 3
    }

    /// pi-ai's deterministic 32-bit hash, evaluated over UTF-16 code units.
    private static func shortHash(_ value: String) -> String {
        var h1: UInt32 = 0xDEAD_BEEF
        var h2: UInt32 = 0x41C6_CE57
        for codeUnit in value.utf16 {
            let character = UInt32(codeUnit)
            h1 = (h1 ^ character) &* 2_654_435_761
            h2 = (h2 ^ character) &* 1_597_334_677
        }
        h1 = ((h1 ^ (h1 >> 16)) &* 2_246_822_507)
            ^ ((h2 ^ (h2 >> 13)) &* 3_266_489_909)
        h2 = ((h2 ^ (h2 >> 16)) &* 2_246_822_507)
            ^ ((h1 ^ (h1 >> 13)) &* 3_266_489_909)
        return String(h2, radix: 36) + String(h1, radix: 36)
    }
}

private struct ToolAssociation {
    var sourceId: String
    var sourceItemId: String?
    var destinationId: String
    var toolName: String
    var providerExecuted: Bool
    var exactMatch: Bool
    var hasResult = false
}
