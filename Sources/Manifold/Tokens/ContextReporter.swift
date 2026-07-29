import Foundation

/// Attributes a request's tokens to the parts of the request that caused them.
///
/// Attribution is provider-independent: it walks a ``CallOptions`` and counts
/// each part. Only the counting is provider-specific, and that is injected as a
/// ``Tokenizer`` — which is why this lives beside the spec rather than inside a
/// wire implementation.
public struct ContextReporter: Sendable {

    public var tokenizer: any Tokenizer

    /// Per-message framing overhead.
    ///
    /// Every provider wraps a message in role markers and delimiters that cost
    /// tokens but appear in no field. Ignoring it under-counts conversations
    /// with many short turns — exactly the shape a long agent session takes.
    public var perMessageOverhead: Int

    public init(tokenizer: any Tokenizer = HeuristicTokenizer(), perMessageOverhead: Int = 4) {
        self.tokenizer = tokenizer
        self.perMessageOverhead = perMessageOverhead
    }

    /// Breaks a request down by category.
    ///
    /// - Parameters:
    ///   - options: the request to account for.
    ///   - contextWindow: the model's window. Pass
    ///     `ProviderCatalog.model(id)?.1.contextWindow` to read it from the
    ///     catalog rather than hardcoding.
    ///   - extras: additional named segments the host wants shown — skills,
    ///     memory files, retrieved documents.
    public func report(
        _ options: CallOptions,
        contextWindow: Int? = nil,
        extras: [(String, Int)] = []
    ) -> ContextUsage {
        var entries: [ContextUsage.Entry] = []

        let systemTokens = options.prompt
            .filter { $0.role == .system }
            .reduce(0) { $0 + count($1) }
        if systemTokens > 0 {
            entries.append(.init(segment: .systemPrompt, tokens: systemTokens))
        }

        let messageTokens = options.prompt
            .filter { $0.role != .system }
            .reduce(0) { $0 + count($1) }
        if messageTokens > 0 {
            entries.append(.init(segment: .messages, tokens: messageTokens))
        }

        let toolTokens = options.tools.reduce(0) { $0 + count($1) }
        if toolTokens > 0 {
            entries.append(.init(segment: .tools, tokens: toolTokens))
        }

        for (name, tokens) in extras where tokens > 0 {
            entries.append(.init(segment: .custom(name), tokens: tokens))
        }

        return ContextUsage(
            entries: entries,
            contextWindow: contextWindow,
            isEstimated: true
        )
    }

    /// Breaks a request down and anchors the total to a known-exact figure.
    ///
    /// The natural source is the previous response's ``Usage/inputTokens``,
    /// which is authoritative and already paid for. This gives an exact total
    /// and proportionally-correct segments without a single extra request.
    public func report(
        _ options: CallOptions,
        contextWindow: Int? = nil,
        extras: [(String, Int)] = [],
        calibratingTo actualTotal: Int
    ) -> ContextUsage {
        report(options, contextWindow: contextWindow, extras: extras)
            .calibrated(toTotal: actualTotal)
    }

    // MARK: - Counting

    private func count(_ message: Message) -> Int {
        message.content.reduce(perMessageOverhead) { $0 + count($1) }
    }

    private func count(_ part: ContentPart) -> Int {
        switch part {
        case .text(let text):
            tokenizer.count(text)

        case .reasoning(let text, _):
            tokenizer.count(text)

        case .toolCall(let call):
            // The serialized arguments are what reaches the model, plus the
            // name and the call's own framing.
            tokenizer.count(call.toolName) + tokenizer.count(call.input) + perMessageOverhead

        case .toolResult(let result):
            tokenizer.count(result.toolName)
                + tokenizer.count((try? result.result.encodedString()) ?? "")
                + perMessageOverhead

        case .file(let file):
            // Images and documents are not text and cannot be estimated from
            // their bytes: providers price them by dimensions or page count,
            // and a base64 blob's length says nothing useful. Counting the
            // encoded string would be wildly wrong, so this contributes only
            // its framing. Pass a measured figure via `extras` when it matters.
            switch file.data {
            case .url(let url): tokenizer.count(url)
            case .base64: perMessageOverhead
            }
        }
    }

    private func count(_ tool: ToolDefinition) -> Int {
        // Schemas are serialized into the request in full, and are routinely
        // larger than the description. A tool surface is often the quietest
        // consumer of a context window.
        tokenizer.count(tool.name)
            + tokenizer.count(tool.description ?? "")
            + tokenizer.count((try? tool.inputSchema.encodedString()) ?? "")
    }
}
