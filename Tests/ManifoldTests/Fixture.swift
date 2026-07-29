import Foundation
import Testing

@testable import Manifold

/// Loads recorded provider responses vendored from the AI SDK.
///
/// See `Tests/ManifoldTests/Fixtures/anthropic/PROVENANCE.md`.
enum Fixture {
    struct LoadError: Error, CustomStringConvertible {
        let description: String
    }

    static var anthropicDirectory: URL {
        get throws {
            guard let resources = Bundle.module.resourceURL else {
                throw LoadError(description: "test bundle has no resource URL")
            }
            let directory = resources.appending(path: "Fixtures/anthropic")
            guard FileManager.default.fileExists(atPath: directory.path) else {
                throw LoadError(description: "missing fixtures at \(directory.path)")
            }
            return directory
        }
    }

    /// Names of every recorded Anthropic stream, sorted for stable test output.
    static var anthropicStreamNames: [String] {
        get throws {
            let files = try FileManager.default.contentsOfDirectory(
                at: try anthropicDirectory,
                includingPropertiesForKeys: nil
            )
            return files
                .filter { $0.lastPathComponent.hasSuffix(".chunks.txt") }
                .map { $0.lastPathComponent.replacingOccurrences(of: ".chunks.txt", with: "") }
                .sorted()
        }
    }

    /// Decoded chunks for one recorded stream, in the order the provider sent them.
    static func anthropicChunks(_ name: String) throws -> [JSONValue] {
        let url = try anthropicDirectory.appending(path: "\(name).chunks.txt")
        let contents = try String(contentsOf: url, encoding: .utf8)

        return try contents
            .split(separator: "\n", omittingEmptySubsequences: true)
            .map { try JSONValue.decode(from: String($0)) }
    }

    /// Splits a recorded file into individual message streams.
    ///
    /// Some fixtures record several sequential API calls into one file — a
    /// programmatic-tool-calling loop, for instance, or a tool search that
    /// takes two round trips. Each `message_start` begins a new stream, and a
    /// mapper instance is per-stream, so they must be replayed separately.
    static func anthropicMessages(_ name: String) throws -> [[JSONValue]] {
        var messages: [[JSONValue]] = []
        var current: [JSONValue] = []

        for chunk in try anthropicChunks(name) {
            if chunk["type"]?.stringValue == "message_start", !current.isEmpty {
                messages.append(current)
                current = []
            }
            current.append(chunk)
        }
        if !current.isEmpty { messages.append(current) }

        return messages
    }

    /// Replays every message in a recorded file, one fresh mapper each.
    static func replayAnthropic(_ name: String) throws -> [[StreamPart]] {
        try anthropicMessages(name).map { chunks in
            var wire = AnthropicMessagesWire()
            return chunks.flatMap { wire.map(chunk: $0) }
        }
    }
}
