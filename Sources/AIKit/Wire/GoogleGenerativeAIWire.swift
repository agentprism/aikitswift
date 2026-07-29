import Foundation

/// Maps Google's Generative AI streaming protocol (Gemini) onto ``StreamPart``.
///
/// Gemini differs from the other two protocols in ways that shape this mapper:
///
/// - **Tool arguments arrive whole.** A `functionCall` part carries its complete
///   `args` object in one chunk rather than as JSON fragments, so the whole
///   input triad is emitted at once.
/// - **Tool calls carry no id.** One is synthesized, deterministically, so the
///   same recording always produces the same ids.
/// - **Reasoning is a flag, not a block.** A thought is an ordinary text part
///   marked `thought: true`, so block transitions have to be inferred.
public struct GoogleGenerativeAIWire: WireMapper {

    private var textOpen: Set<String> = []
    private var reasoningOpen: Set<String> = []

    private var finishReason: FinishReason = .other
    private var usage: JSONValue?
    private var emittedStreamStart = false
    private var emittedMetadata = false
    private var finished = false
    /// Makes synthesized tool call ids deterministic.
    private var toolCallCounter = 0
    /// A `codeExecutionResult` part carries no id, so it is paired with the
    /// `executableCode` that preceded it.
    private var lastCodeExecutionId: String?
    /// Grounding metadata repeats on every chunk; sources are emitted once.
    private var emittedSources: Set<String> = []

    public init() {}

    // MARK: - Entry points

    public mutating func map(chunk: JSONValue) -> [StreamPart] {
        var parts: [StreamPart] = []

        if !emittedStreamStart {
            emittedStreamStart = true
            parts.append(.streamStart(warnings: []))
        }

        parts.append(contentsOf: mapChunk(chunk))
        return parts
    }

    public mutating func map(rawJSON: String) -> [StreamPart] {
        let trimmed = rawJSON.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed != "[DONE]" else { return finish() }

        do {
            return map(chunk: try JSONValue.decode(from: trimmed))
        } catch {
            return [.error(StreamError(
                type: "parse_error",
                message: "Failed to decode chunk: \(error)",
                raw: .string(rawJSON)
            ))]
        }
    }

    /// Closes the stream.
    ///
    /// Gemini has no terminal event — the connection simply ends — so the
    /// final `finish` exists only because this is called.
    public mutating func finish() -> [StreamPart] {
        guard !finished else { return [] }
        finished = true

        var parts: [StreamPart] = []

        if !emittedStreamStart {
            emittedStreamStart = true
            parts.insert(.streamStart(warnings: []), at: 0)
        }

        parts.append(contentsOf: closeOpenBlocks())
        parts.append(.finish(
            usage: usage.map(Self.convertUsage) ?? .empty,
            finishReason: finishReason,
            providerMetadata: usage.map { ["google": ["usageMetadata": $0]] }
        ))

        return parts
    }

    // MARK: - Chunk handling

    private mutating func mapChunk(_ chunk: JSONValue) -> [StreamPart] {
        var parts: [StreamPart] = []

        // Usage is cumulative and repeated on every chunk, so the last one wins.
        if let metadata = chunk["usageMetadata"], !metadata.isNull {
            usage = metadata
        }

        if !emittedMetadata, chunk["responseId"] != nil || chunk["modelVersion"] != nil {
            emittedMetadata = true
            parts.append(.responseMetadata(ResponseMetadata(
                id: chunk["responseId"]?.stringValue,
                modelId: chunk["modelVersion"]?.stringValue
            )))
        }

        guard let candidates = chunk["candidates"]?.arrayValue, !candidates.isEmpty else {
            // A usage-only trailer, or a prompt-feedback chunk. Usage was
            // captured above, so nothing is lost.
            return parts
        }

        for candidate in candidates {
            parts.append(contentsOf: mapCandidate(candidate))
        }

        return parts
    }

    private mutating func mapCandidate(_ candidate: JSONValue) -> [StreamPart] {
        var parts: [StreamPart] = []
        let id = String(candidate["index"]?.intValue ?? 0)

        for part in candidate["content"]?["parts"]?.arrayValue ?? [] {
            parts.append(contentsOf: mapPart(part, id: id))
        }

        // Grounding is Gemini's search citation channel. It sits on the
        // candidate rather than in `parts`, and repeats on every chunk, so
        // sources are deduplicated by URL.
        for chunk in candidate["groundingMetadata"]?["groundingChunks"]?.arrayValue ?? [] {
            guard let web = chunk["web"],
                  let uri = web["uri"]?.stringValue,
                  emittedSources.insert(uri).inserted else { continue }

            parts.append(.source(Source(
                id: "grounding-\(emittedSources.count)",
                kind: .url(uri),
                title: web["title"]?.stringValue
            )))
        }

        if let reason = candidate["finishReason"]?.stringValue {
            finishReason = Self.mapFinishReason(reason)
            parts.append(contentsOf: closeOpenBlocks())
        }

        return parts
    }

    private mutating func mapPart(_ part: JSONValue, id: String) -> [StreamPart] {
        // A function call carries its complete arguments, so the input triad
        // opens and closes in one step.
        if let call = part["functionCall"] {
            return mapFunctionCall(call, signature: part["thoughtSignature"])
        }

        // Gemini runs code server-side and reports it as two sibling parts:
        // the source it decided to run, then the outcome. They are normalized
        // as a provider-executed call and its result so a caller handles them
        // the same way as any other server tool — and, critically, does not
        // try to execute the code itself.
        if let executable = part["executableCode"] {
            toolCallCounter += 1
            let callId = "code_execution-\(toolCallCounter)"
            lastCodeExecutionId = callId
            let input = (try? executable.encodedString()) ?? "{}"

            return [
                .toolInputStart(id: callId, toolName: "code_execution", providerExecuted: true),
                .toolInputDelta(id: callId, delta: input),
                .toolInputEnd(id: callId),
                .toolCall(ToolCall(
                    toolCallId: callId,
                    toolName: "code_execution",
                    input: input,
                    providerExecuted: true
                )),
            ]
        }

        if let result = part["codeExecutionResult"] {
            // The result arrives as its own part and carries no id, so it is
            // paired with the most recent execution.
            guard let callId = lastCodeExecutionId else { return [.raw(part)] }
            return [.toolResult(ToolResult(
                toolCallId: callId,
                toolName: "code_execution",
                result: result,
                isError: result["outcome"]?.stringValue.map { $0 != "OUTCOME_OK" } ?? false,
                providerMetadata: ["google": ["partType": .string("codeExecutionResult")]]
            ))]
        }

        if let inline = part["inlineData"] {
            guard let mediaType = inline["mimeType"]?.stringValue,
                  let data = inline["data"]?.stringValue else {
                return [.raw(part)]
            }
            return [.file(GeneratedFile(mediaType: mediaType, data: .base64(data)))]
        }

        guard let text = part["text"]?.stringValue else {
            // Executable code, code results, and whatever Gemini adds next.
            return [.raw(part)]
        }

        // Reasoning is a flag on an ordinary text part rather than a distinct
        // block, so a switch between the two has to be detected here.
        let isThought = part["thought"]?.boolValue == true
        guard !text.isEmpty else { return [] }

        var parts: [StreamPart] = []

        if isThought {
            if textOpen.contains(id) {
                textOpen.remove(id)
                parts.append(.textEnd(id: id))
            }
            if !reasoningOpen.contains(id) {
                reasoningOpen.insert(id)
                parts.append(.reasoningStart(id: id))
            }
            parts.append(.reasoningDelta(id: id, delta: text))
        } else {
            if reasoningOpen.contains(id) {
                reasoningOpen.remove(id)
                parts.append(.reasoningEnd(id: id))
            }
            if !textOpen.contains(id) {
                textOpen.insert(id)
                parts.append(.textStart(id: id))
            }
            parts.append(.textDelta(id: id, delta: text))
        }

        return parts
    }

    private mutating func mapFunctionCall(_ call: JSONValue, signature: JSONValue?) -> [StreamPart] {
        guard let name = call["name"]?.stringValue else { return [.raw(call)] }

        // Gemini assigns no call id. One is synthesized from a counter so that
        // replaying the same stream twice yields identical ids — otherwise
        // nothing downstream could be compared or cached.
        toolCallCounter += 1
        let callId = "\(name)-\(toolCallCounter)"

        let arguments = (try? (call["args"] ?? .object([:])).encodedString()) ?? "{}"

        // The signature must be echoed back unchanged on the next turn for
        // multi-turn reasoning to work, so it is preserved rather than dropped.
        var metadata: ProviderMetadata?
        if let signature, !signature.isNull {
            metadata = ["google": ["thoughtSignature": signature]]
        }

        return [
            .toolInputStart(id: callId, toolName: name),
            .toolInputDelta(id: callId, delta: arguments),
            .toolInputEnd(id: callId),
            .toolCall(ToolCall(
                toolCallId: callId,
                toolName: name,
                input: arguments,
                providerMetadata: metadata
            )),
        ]
    }

    private mutating func closeOpenBlocks() -> [StreamPart] {
        var parts: [StreamPart] = []

        for id in reasoningOpen.sorted() { parts.append(.reasoningEnd(id: id)) }
        reasoningOpen.removeAll()

        for id in textOpen.sorted() { parts.append(.textEnd(id: id)) }
        textOpen.removeAll()

        return parts
    }

    // MARK: - Mapping tables

    static func mapFinishReason(_ raw: String?) -> FinishReason {
        let unified: FinishReason.Unified
        switch raw {
        case "STOP": unified = .stop
        case "MAX_TOKENS": unified = .length
        case "SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII", "IMAGE_SAFETY":
            unified = .contentFilter
        case "MALFORMED_FUNCTION_CALL": unified = .error
        default: unified = .other
        }
        return FinishReason(unified: unified, raw: raw)
    }

    /// Converts Gemini's `usageMetadata` into normalized ``Usage``.
    ///
    /// - Important: `candidatesTokenCount` **excludes** thinking tokens — the
    ///   opposite of OpenAI, where `completion_tokens` includes them. The
    ///   provider's own `totalTokenCount` confirms it: prompt + candidates +
    ///   thoughts. Treating `candidatesTokenCount` as the output total silently
    ///   under-reports every reasoning call, often by an order of magnitude.
    static func convertUsage(_ usage: JSONValue) -> Usage {
        let promptTokens = usage["promptTokenCount"]?.intValue
        let candidateTokens = usage["candidatesTokenCount"]?.intValue ?? 0
        let thoughtTokens = usage["thoughtsTokenCount"]?.intValue
        let cachedTokens = usage["cachedContentTokenCount"]?.intValue ?? 0

        return Usage(
            inputTokens: .init(
                total: promptTokens,
                noCache: promptTokens.map { $0 - cachedTokens },
                cacheRead: cachedTokens,
                // Gemini's implicit caching is not billed as a write.
                cacheWrite: 0
            ),
            outputTokens: .init(
                total: candidateTokens + (thoughtTokens ?? 0),
                text: candidateTokens,
                reasoning: thoughtTokens
            ),
            raw: usage
        )
    }
}
