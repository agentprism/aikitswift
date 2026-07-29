import Foundation
import Testing

@testable import Manifold

/// Loads recorded provider responses vendored from the AI SDK.
///
/// See `Tests/ManifoldTests/Fixtures/PROVENANCE.md`.
enum Fixture {
    struct LoadError: Error, CustomStringConvertible {
        let description: String
    }

    static func directory(_ set: String) throws -> URL {
        guard let resources = Bundle.module.resourceURL else {
            throw LoadError(description: "test bundle has no resource URL")
        }
        let directory = resources.appending(path: "Fixtures/\(set)")
        guard FileManager.default.fileExists(atPath: directory.path) else {
            throw LoadError(description: "missing fixture set at \(directory.path)")
        }
        return directory
    }

    /// Names of every recorded stream in a set, sorted for stable test output.
    static func streamNames(_ set: String) throws -> [String] {
        try FileManager.default
            .contentsOfDirectory(at: try directory(set), includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasSuffix(".chunks.txt") }
            .map { $0.lastPathComponent.replacingOccurrences(of: ".chunks.txt", with: "") }
            .sorted()
    }

    /// Decoded chunks for one recording, in the order the provider sent them.
    ///
    /// Lines that are not JSON — the `[DONE]` sentinel, for instance — are
    /// skipped here and exercised separately at the mapper's string entry point.
    static func chunks(_ set: String, _ name: String) throws -> [JSONValue] {
        let url = try directory(set).appending(path: "\(name).chunks.txt")
        let contents = try String(contentsOf: url, encoding: .utf8)

        return contents
            .split(separator: "\n", omittingEmptySubsequences: true)
            .compactMap { try? JSONValue.decode(from: String($0)) }
    }

    /// Splits a recording into individual message streams.
    ///
    /// Some recordings hold several sequential API calls in one file — a
    /// programmatic-tool-calling loop, or a tool search that needs two round
    /// trips. A mapper instance is per-stream, so they must be replayed apart.
    static func streams(
        _ set: String,
        _ name: String,
        splitOn isBoundary: (@Sendable (JSONValue) -> Bool)? = nil
    ) throws -> [[JSONValue]] {
        let chunks = try chunks(set, name)
        guard let isBoundary else { return chunks.isEmpty ? [] : [chunks] }

        var streams: [[JSONValue]] = []
        var current: [JSONValue] = []

        for chunk in chunks {
            if isBoundary(chunk), !current.isEmpty {
                streams.append(current)
                current = []
            }
            current.append(chunk)
        }
        if !current.isEmpty { streams.append(current) }

        return streams
    }

    /// Replays every stream in a recording, one fresh mapper each.
    ///
    /// `finish()` is always called: it is idempotent, and several protocols
    /// carry final usage nowhere else.
    static func replay<M: WireMapper>(
        _ mapper: M.Type,
        _ set: String,
        _ name: String,
        splitOn isBoundary: (@Sendable (JSONValue) -> Bool)? = nil
    ) throws -> [[StreamPart]] {
        try streams(set, name, splitOn: isBoundary).map { chunks in
            var wire = M()
            var parts = chunks.flatMap { wire.map(chunk: $0) }
            parts.append(contentsOf: wire.finish())
            return parts
        }
    }

    /// Splits where a new Anthropic message begins.
    static let anthropicBoundary: @Sendable (JSONValue) -> Bool = {
        $0["type"]?.stringValue == "message_start"
    }
}
