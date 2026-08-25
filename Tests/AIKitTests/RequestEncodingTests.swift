import Foundation
import Testing

@testable import AIKit

@Suite("Request encoding")
struct RequestEncodingTests {

    static let conversation: Prompt = [
        .system("You are terse."),
        .user("Weather in Paris?"),
        Message(role: .assistant, content: [.toolCall(ToolCall(
            toolCallId: "call_1", toolName: "get_weather", input: #"{"city":"Paris"}"#
        ))]),
        .toolResult(toolCallId: "call_1", toolName: "get_weather", result: "18C"),
    ]

    static let weatherTool = ToolDefinition(
        name: "get_weather",
        description: "Current weather for a city",
        inputSchema: ["type": "object", "properties": ["city": ["type": "string"]]]
    )

    // MARK: - Anthropic

    @Test("Anthropic lifts system out of the message list")
    func anthropicLiftsSystem() {
        let body = AnthropicMessagesRequest.encode(
            CallOptions(model: "claude-opus-4-8", prompt: Self.conversation)
        ).body

        #expect(body["system"]?.stringValue == "You are terse.")
        // A system *message* would be rejected; it must be the top-level field.
        let roles = (body["messages"]?.arrayValue ?? []).compactMap { $0["role"]?.stringValue }
        #expect(!roles.contains("system"))
    }

    @Test("Anthropic always sends max_tokens")
    func anthropicAlwaysSendsMaxTokens() {
        // Required by this API alone; omitting it is a 400.
        let body = AnthropicMessagesRequest.encode(
            CallOptions(model: "claude-opus-4-8", prompt: [.user("hi")])
        ).body

        #expect((body["max_tokens"]?.intValue ?? 0) > 0)
    }

    @Test("Anthropic carries tool results in a user turn")
    func anthropicPutsToolResultsInUserTurn() {
        // There is no `tool` role here — results are blocks in a user message.
        let body = AnthropicMessagesRequest.encode(
            CallOptions(model: "claude-opus-4-8", prompt: Self.conversation)
        ).body

        let last = body["messages"]?.arrayValue?.last
        #expect(last?["role"]?.stringValue == "user")
        #expect(last?["content"]?[0]?["type"]?.stringValue == "tool_result")
        #expect(last?["content"]?[0]?["tool_use_id"]?.stringValue == "call_1")
    }

    @Test("Anthropic drops temperature for models that reject it")
    func anthropicDropsTemperature() {
        // Newer Anthropic models reject sampling parameters with a 400 rather
        // than ignoring them, so this turns a hard failure into a warning.
        let model = ModelInfo(id: "claude-opus-4-8", temperature: false)
        let encoded = AnthropicMessagesRequest.encode(
            CallOptions(model: "claude-opus-4-8", prompt: [.user("hi")], temperature: 0.7),
            model: model
        )

        #expect(encoded.body["temperature"] == nil)
        #expect(encoded.warnings.contains { $0.setting == "temperature" })
    }

    @Test("Anthropic keeps temperature for models that accept it")
    func anthropicKeepsTemperature() {
        let encoded = AnthropicMessagesRequest.encode(
            CallOptions(model: "old", prompt: [.user("hi")], temperature: 0.7),
            model: ModelInfo(id: "old", temperature: true)
        )

        #expect(encoded.body["temperature"]?.doubleValue == 0.7)
        #expect(encoded.warnings.isEmpty)
    }

    @Test("Anthropic sends tool schemas as input_schema")
    func anthropicNamesSchemaField() {
        let body = AnthropicMessagesRequest.encode(
            CallOptions(model: "claude-opus-4-8", prompt: [.user("hi")], tools: [Self.weatherTool])
        ).body

        #expect(body["tools"]?[0]?["input_schema"] != nil)
        #expect(body["tools"]?[0]?["name"]?.stringValue == "get_weather")
    }

    // MARK: - OpenAI Completions

    @Test("OpenAI opts in to usage on streamed responses")
    func openAIRequestsUsage() {
        // Without `stream_options.include_usage` the API returns no usage at
        // all on a stream — every token count silently lost.
        let body = OpenAICompletionsRequest.encode(
            CallOptions(model: "gpt-5", prompt: [.user("hi")])
        ).body

        #expect(body["stream_options"]?["include_usage"]?.boolValue == true)
    }

    @Test("OpenAI gives each tool result its own message")
    func openAISplitsToolResults() {
        // Unlike Anthropic, which batches results into one user turn.
        let body = OpenAICompletionsRequest.encode(
            CallOptions(model: "gpt-5", prompt: Self.conversation)
        ).body

        let last = body["messages"]?.arrayValue?.last
        #expect(last?["role"]?.stringValue == "tool")
        #expect(last?["tool_call_id"]?.stringValue == "call_1")
        // A string result must not arrive wrapped in JSON quotes.
        #expect(last?["content"]?.stringValue == "18C")
    }

    @Test("OpenAI keeps system as a message")
    func openAIKeepsSystemMessage() {
        let body = OpenAICompletionsRequest.encode(
            CallOptions(model: "gpt-5", prompt: Self.conversation)
        ).body

        #expect(body["messages"]?[0]?["role"]?.stringValue == "system")
    }

    @Test("OpenAI renames the token cap for reasoning models")
    func openAIUsesCompletionTokenCap() {
        // Reasoning models rejected `max_tokens` in favour of
        // `max_completion_tokens`, and the budget covers reasoning too.
        let reasoning = OpenAICompletionsRequest.encode(
            CallOptions(model: "o5", prompt: [.user("hi")], maxOutputTokens: 100),
            model: ModelInfo(id: "o5", reasoning: .init(supported: true))
        ).body
        #expect(reasoning["max_completion_tokens"]?.intValue == 100)
        #expect(reasoning["max_tokens"] == nil)

        let plain = OpenAICompletionsRequest.encode(
            CallOptions(model: "gpt-4", prompt: [.user("hi")], maxOutputTokens: 100),
            model: ModelInfo(id: "gpt-4")
        ).body
        #expect(plain["max_tokens"]?.intValue == 100)
    }

    @Test("OpenAI warns that top_k has no equivalent")
    func openAIWarnsAboutTopK() {
        let encoded = OpenAICompletionsRequest.encode(
            CallOptions(model: "gpt-5", prompt: [.user("hi")], topK: 40)
        )
        #expect(encoded.warnings.contains { $0.setting == "topK" })
    }

    @Test("OpenAI nests tool definitions under function")
    func openAINestsToolDefinitions() {
        let body = OpenAICompletionsRequest.encode(
            CallOptions(model: "gpt-5", prompt: [.user("hi")], tools: [Self.weatherTool])
        ).body

        #expect(body["tools"]?[0]?["type"]?.stringValue == "function")
        #expect(body["tools"]?[0]?["function"]?["parameters"] != nil)
    }

    // MARK: - Google

    @Test("Google renames the assistant role to model")
    func googleRenamesAssistant() {
        let body = GoogleGenerativeAIRequest.encode(
            CallOptions(model: "gemini-3-pro", prompt: Self.conversation)
        ).body

        let roles = (body["contents"]?.arrayValue ?? []).compactMap { $0["role"]?.stringValue }
        #expect(roles.contains("model"))
        #expect(!roles.contains("assistant"))
    }

    @Test("Google puts system instructions in their own field")
    func googleSeparatesSystemInstruction() {
        let body = GoogleGenerativeAIRequest.encode(
            CallOptions(model: "gemini-3-pro", prompt: Self.conversation)
        ).body

        #expect(body["systemInstruction"]?["parts"]?[0]?["text"]?.stringValue == "You are terse.")
    }

    @Test("Google wraps non-object tool results")
    func googleWrapsScalarToolResults() {
        // `functionResponse.response` must be an object; a bare scalar is
        // rejected, so it is wrapped rather than sent raw.
        let body = GoogleGenerativeAIRequest.encode(
            CallOptions(model: "gemini-3-pro", prompt: Self.conversation)
        ).body

        let response = body["contents"]?.arrayValue?.last?["parts"]?[0]?["functionResponse"]?["response"]
        #expect(response?["result"]?.stringValue == "18C")
    }

    @Test("Google declares all functions in one tools entry")
    func googleGroupsFunctionDeclarations() {
        let body = GoogleGenerativeAIRequest.encode(
            CallOptions(model: "gemini-3-pro", prompt: [.user("hi")], tools: [Self.weatherTool])
        ).body

        #expect(body["tools"]?.arrayValue?.count == 1)
        #expect(body["tools"]?[0]?["functionDeclarations"]?.arrayValue?.count == 1)
    }

    @Test("Google sends tool schemas verbatim under parametersJsonSchema")
    func googleSendsJSONSchemaParameters() {
        // `parameters` is an OpenAPI-subset proto with a fixed field list: a
        // schema carrying `additionalProperties` (or any other keyword it never
        // declared) is rejected with `Unknown name "additionalProperties" ...
        // Cannot find field`, and nothing in the request survives it.
        let strictTool = ToolDefinition(
            name: "ask_user",
            description: "Ask one question",
            inputSchema: [
                "type": "object",
                "properties": [
                    "options": [
                        "type": "array",
                        "items": [
                            "type": "object",
                            "properties": ["label": ["type": "string"]],
                            "additionalProperties": false,
                        ],
                    ]
                ],
                "additionalProperties": false,
            ]
        )
        let body = GoogleGenerativeAIRequest.encode(
            CallOptions(model: "gemini-3-pro", prompt: [.user("hi")], tools: [strictTool])
        ).body

        let declaration = body["tools"]?[0]?["functionDeclarations"]?[0]
        #expect(declaration?["parameters"] == nil)
        #expect(declaration?["parametersJsonSchema"]?["additionalProperties"]?.boolValue == false)
        // Nested schemas travel untouched too; the API reports each offending
        // path separately, so a top-level-only fix still fails the request.
        let items = declaration?["parametersJsonSchema"]?["properties"]?["options"]?["items"]
        #expect(items?["additionalProperties"]?.boolValue == false)
    }

    @Test("Google sends tool arguments as an object, not a string")
    func googleSendsArgsAsObject() {
        let body = GoogleGenerativeAIRequest.encode(
            CallOptions(model: "gemini-3-pro", prompt: Self.conversation)
        ).body

        let call = (body["contents"]?.arrayValue ?? [])
            .compactMap { $0["parts"]?[0]?["functionCall"] }
            .first
        #expect(call?["args"]?["city"]?.stringValue == "Paris")
    }

    // MARK: - OpenAI Responses

    @Test("Responses lifts system into instructions")
    func responsesUsesInstructions() {
        let body = OpenAIResponsesRequest.encode(
            CallOptions(model: "gpt-5", prompt: Self.conversation)
        ).body

        #expect(body["instructions"]?.stringValue == "You are terse.")
    }

    @Test("Responses emits tool traffic as input items")
    func responsesEmitsToolItems() {
        // Calls and results are peers of messages here, not fields on them.
        let body = OpenAIResponsesRequest.encode(
            CallOptions(model: "gpt-5", prompt: Self.conversation)
        ).body

        let types = (body["input"]?.arrayValue ?? []).compactMap { $0["type"]?.stringValue }
        #expect(types.contains("function_call"))
        #expect(types.contains("function_call_output"))
    }

    @Test("Responses declares tools flat")
    func responsesDeclaresToolsFlat() {
        // Unlike Completions, where the definition nests under `function`.
        let body = OpenAIResponsesRequest.encode(
            CallOptions(model: "gpt-5", prompt: [.user("hi")], tools: [Self.weatherTool])
        ).body

        #expect(body["tools"]?[0]?["name"]?.stringValue == "get_weather")
        #expect(body["tools"]?[0]?["function"] == nil)
    }

    @Test("Responses warns that stop sequences were dropped")
    func responsesWarnsAboutStopSequences() {
        let encoded = OpenAIResponsesRequest.encode(
            CallOptions(model: "gpt-5", prompt: [.user("hi")], stopSequences: ["END"])
        )
        #expect(encoded.warnings.contains { $0.setting == "stopSequences" })
    }

    // MARK: - Cross-cutting

    @Test("provider options override encoded fields")
    func providerOptionsOverride() {
        // The escape hatch has to actually win, or it is not an escape hatch.
        let options = CallOptions(
            model: "claude-opus-4-8",
            prompt: [.user("hi")],
            providerOptions: ["anthropic": ["thinking": ["type": "adaptive"], "max_tokens": 99]]
        )
        let body = AnthropicMessagesRequest.encode(options).body

        #expect(body["thinking"]?["type"]?.stringValue == "adaptive")
        #expect(body["max_tokens"]?.intValue == 99)
    }

    @Test("bodies serialize with sorted keys")
    func serializesDeterministically() throws {
        // Prompt caching is a byte-level prefix match, so unstable key order
        // silently destroys every cache hit.
        let body = AnthropicMessagesRequest.encode(
            CallOptions(model: "claude-opus-4-8", prompt: [.user("hi")])
        ).body

        #expect(try body.encodedString() == (try body.encodedString()))
        #expect(try body.encodedString().contains("\"max_tokens\""))
    }
}
