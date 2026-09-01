import Foundation

/// Decodes a complete (non-streamed) response body into normalized parts.
///
/// Rather than reimplementing each protocol a second time, a complete body is
/// **replayed as the event sequence a stream would have produced** and fed
/// through the same mapper. Streaming and non-streaming therefore cannot drift:
/// every fix to a mapper lands in both paths, and the fixture suite covers both
/// with one implementation.
///
/// The output is a `[StreamPart]` sequence identical in shape to a stream, so
/// downstream code does not branch on how the response was fetched.
public enum NonStreamingResponse {

    /// Decodes a complete response body.
    public static func decode(_ body: JSONValue, wire: WireProtocol) -> [StreamPart] {
        var mapper = wire.makeMapper()
        var parts = synthesize(body, wire: wire).flatMap { mapper.map(chunk: $0) }
        parts.append(contentsOf: mapper.finish())
        return parts
    }

    /// Rewrites a complete body as the chunks a stream would have delivered.
    static func synthesize(_ body: JSONValue, wire: WireProtocol) -> [JSONValue] {
        switch wire {
        case .anthropicMessages: anthropic(body)
        case .openAICompletions: openAICompletions(body)
        case .openAIResponses, .openAICodex: openAIResponses(body)
        // Gemini's non-streaming body already has the shape of a stream chunk,
        // so it needs no rewriting at all.
        case .googleGenerativeAI: [body]
        }
    }

    // MARK: - Anthropic

    private static func anthropic(_ body: JSONValue) -> [JSONValue] {
        var chunks: [JSONValue] = []

        // `message_start` carries input and cache counts; the output counts
        // arrive on `message_delta`. Splitting them mirrors the stream so the
        // mapper's two-phase usage accumulation runs exactly as it would live.
        var startUsage: [String: JSONValue] = [:]
        for key in ["input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"] {
            if let value = body["usage"]?[key] { startUsage[key] = value }
        }

        var message: [String: JSONValue] = ["usage": .object(startUsage)]
        for key in ["id", "model", "role", "type"] {
            if let value = body[key] { message[key] = value }
        }
        chunks.append(.object(["type": "message_start", "message": .object(message)]))

        for (index, block) in (body["content"]?.arrayValue ?? []).enumerated() {
            chunks.append(contentsOf: anthropicBlock(block, index: index))
        }

        var delta: [String: JSONValue] = [:]
        for key in ["stop_reason", "stop_sequence", "stop_details"] {
            if let value = body[key] { delta[key] = value }
        }
        chunks.append(.object([
            "type": "message_delta",
            "delta": .object(delta),
            "usage": body["usage"] ?? .object([:]),
        ]))

        chunks.append(.object(["type": "message_stop"]))
        return chunks
    }

    private static func anthropicBlock(_ block: JSONValue, index: Int) -> [JSONValue] {
        let position = JSONValue.number(Double(index))
        var chunks: [JSONValue] = []

        switch block["type"]?.stringValue {
        case "text":
            // The opening block is emptied and the text replayed as a delta,
            // matching how it actually arrives.
            chunks.append(.object([
                "type": "content_block_start",
                "index": position,
                "content_block": .object(["type": "text", "text": .string("")]),
            ]))
            if let text = block["text"]?.stringValue, !text.isEmpty {
                chunks.append(.object([
                    "type": "content_block_delta",
                    "index": position,
                    "delta": .object(["type": "text_delta", "text": .string(text)]),
                ]))
            }

        case "thinking":
            chunks.append(.object([
                "type": "content_block_start",
                "index": position,
                // Retain the complete provider block as metadata in the mapper;
                // deltas below still produce the normalized readable content.
                "content_block": block,
            ]))
            if let thinking = block["thinking"]?.stringValue, !thinking.isEmpty {
                chunks.append(.object([
                    "type": "content_block_delta",
                    "index": position,
                    "delta": .object(["type": "thinking_delta", "thinking": .string(thinking)]),
                ]))
            }
            // The signature must survive: it is replayed back to the API
            // verbatim on the next turn or the request is rejected.
            if let signature = block["signature"] {
                chunks.append(.object([
                    "type": "content_block_delta",
                    "index": position,
                    "delta": .object(["type": "signature_delta", "signature": signature]),
                ]))
            }

        default:
            // Tool calls, server tool results, redacted thinking and compaction
            // all arrive complete; the mapper already reads them from the
            // opening block.
            chunks.append(.object([
                "type": "content_block_start",
                "index": position,
                "content_block": block,
            ]))
        }

        chunks.append(.object(["type": "content_block_stop", "index": position]))
        return chunks
    }

    // MARK: - OpenAI Chat Completions

    private static func openAICompletions(_ body: JSONValue) -> [JSONValue] {
        var chunks: [JSONValue] = []

        for choice in body["choices"]?.arrayValue ?? [] {
            let index = choice["index"] ?? .number(0)
            guard let message = choice["message"] else { continue }

            // A complete message becomes a single delta carrying everything.
            var delta: [String: JSONValue] = [:]
            for key in ["content", "reasoning_content", "reasoning", "tool_calls"] {
                if let value = message[key], !value.isNull { delta[key] = value }
            }

            var chunk: [String: JSONValue] = [
                "choices": .array([.object(["index": index, "delta": .object(delta)])])
            ]
            for key in ["id", "model", "created"] {
                if let value = body[key] { chunk[key] = value }
            }
            chunks.append(.object(chunk))

            chunks.append(.object([
                "choices": .array([.object([
                    "index": index,
                    "delta": .object([:]),
                    "finish_reason": choice["finish_reason"] ?? .string("stop"),
                ])])
            ]))
        }

        if let usage = body["usage"], !usage.isNull {
            chunks.append(.object(["choices": .array([]), "usage": usage]))
        }

        return chunks
    }

    // MARK: - OpenAI Responses

    private static func openAIResponses(_ body: JSONValue) -> [JSONValue] {
        var chunks: [JSONValue] = [.object(["type": "response.created", "response": body])]

        for (outputIndex, item) in (body["output"]?.arrayValue ?? []).enumerated() {
            // `id` is optional on Responses input/output items. The synthetic
            // event stream still needs a stable internal key, but that key
            // must never be written into the retained item itself.
            let itemId = item["id"]?.stringValue ?? "output-index:\(outputIndex)"
            let position = JSONValue.number(Double(outputIndex))
            chunks.append(.object([
                "type": "response.output_item.added",
                "output_index": position,
                "item": item,
            ]))

            switch item["type"]?.stringValue {
            case "message":
                for (contentIndex, part) in (item["content"]?.arrayValue ?? []).enumerated() {
                    let position = JSONValue.number(Double(contentIndex))
                    guard part["type"]?.stringValue == "output_text" else { continue }

                    chunks.append(.object([
                        "type": "response.content_part.added",
                        "item_id": .string(itemId),
                        "output_index": .number(Double(outputIndex)),
                        "content_index": position,
                        "part": .object([
                            "type": "output_text",
                            "text": "",
                            "annotations": [],
                        ]),
                    ]))
                    if let text = part["text"]?.stringValue, !text.isEmpty {
                        chunks.append(.object([
                            "type": "response.output_text.delta",
                            "item_id": .string(itemId),
                            "output_index": .number(Double(outputIndex)),
                            "content_index": position,
                            "delta": .string(text),
                        ]))
                    }
                    chunks.append(.object([
                        "type": "response.content_part.done",
                        "item_id": .string(itemId),
                        "output_index": .number(Double(outputIndex)),
                        "content_index": position,
                        "part": part,
                    ]))
                }

            case "function_call":
                if let arguments = item["arguments"]?.stringValue, !arguments.isEmpty {
                    chunks.append(.object([
                            "type": "response.function_call_arguments.delta",
                            "item_id": .string(itemId),
                            "output_index": position,
                            "delta": .string(arguments),
                    ]))
                }

            case "reasoning":
                for (summaryIndex, summary) in (item["summary"]?.arrayValue ?? []).enumerated() {
                    guard let text = summary["text"]?.stringValue, !text.isEmpty else { continue }
                    chunks.append(.object([
                        "type": "response.reasoning_summary_text.delta",
                        "item_id": .string(itemId),
                        "output_index": position,
                        "summary_index": .number(Double(summaryIndex)),
                        "delta": .string(text),
                    ]))
                }

            default:
                break
            }

            chunks.append(.object([
                "type": "response.output_item.done",
                "output_index": position,
                "item": item,
            ]))
        }

        let terminal = switch body["status"]?.stringValue {
        case "incomplete": "response.incomplete"
        case "failed": "response.failed"
        default: "response.completed"
        }
        chunks.append(.object(["type": .string(terminal), "response": body]))

        return chunks
    }
}
