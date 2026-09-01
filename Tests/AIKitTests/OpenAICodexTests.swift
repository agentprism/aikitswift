import Foundation
import Testing

@testable import AIKit

@Suite("OpenAI Codex", .serialized)
struct OpenAICodexTests {
    private let options = CallOptions(model: "gpt-5-codex", prompt: [.user("hello")])

    @Test("Codex and ordinary Responses select distinct endpoints")
    func selectsCodexEndpointOnlyForCodex() {
        let client = AIClient(
            provider: .init(id: "codex", api: nil, speaking: .openAICodex),
            configuration: .init()
        )
        let backend = URL(string: "https://chatgpt.com/backend-api")!
        let ordinary = URL(string: "https://api.openai.com")!

        #expect(
            client.endpoint(wire: .openAICodex, base: backend, model: "m").absoluteString
                == "https://chatgpt.com/backend-api/codex/responses"
        )
        #expect(
            client.endpoint(
                wire: .openAICodex,
                base: URL(string: "https://chatgpt.com/backend-api/codex/responses")!,
                model: "m"
            ).absoluteString == "https://chatgpt.com/backend-api/codex/responses"
        )
        #expect(
            client.endpoint(wire: .openAIResponses, base: ordinary, model: "m").path
                == "/v1/responses"
        )
    }

    @Test("Codex request defaults and fields are explicit")
    func encodesCodexRequest() {
        let encoded = OpenAICodexResponsesRequest.encode(CallOptions(
            model: "gpt-5-codex",
            prompt: [.system("Use tools safely."), .user("hello")],
            maxOutputTokens: 222,
            tools: [ToolDefinition(
                name: "weather",
                description: "Get weather",
                inputSchema: ["type": "object"],
                strict: true
            )],
            toolChoice: .tool(name: "weather"),
            thinking: .level(.high),
            providerOptions: ["openai-codex": ["prompt_cache_key": "session-1"]]
        )).body

        #expect(encoded["model"]?.stringValue == "gpt-5-codex")
        #expect(encoded["instructions"]?.stringValue == "Use tools safely.")
        #expect(encoded["store"]?.boolValue == false)
        #expect(encoded["stream"]?.boolValue == true)
        #expect(encoded["parallel_tool_calls"]?.boolValue == true)
        #expect(encoded["text"]?["verbosity"]?.stringValue == "low")
        #expect(encoded["include"]?.arrayValue == ["reasoning.encrypted_content"])
        #expect(encoded["max_output_tokens"]?.intValue == 222)
        #expect(encoded["reasoning"]?["effort"]?.stringValue == "high")
        #expect(encoded["reasoning"]?["summary"]?.stringValue == "auto")
        #expect(encoded["prompt_cache_key"]?.stringValue == "session-1")
        #expect(encoded["tool_choice"]?["name"]?.stringValue == "weather")
        #expect(encoded["tools"]?[0]?["strict"]?.boolValue == true)
        #expect(encoded["input"]?[0]?["role"]?.stringValue == "user")
    }

    @Test("Codex encoding stays streaming when shared dispatch requests non-streaming")
    func codexEncodingCannotDisableStreaming() {
        let client = AIClient(
            provider: .init(id: "codex", api: nil, speaking: .openAICodex),
            configuration: .init()
        )

        let encoded = client.encode(
            options,
            wire: .openAICodex,
            model: nil,
            streaming: false
        )

        #expect(encoded.body["stream"]?.boolValue == true)
    }

    @Test("Codex strict defaults preserve explicit strict tools, including namespaces")
    func preservesExplicitStrictTools() {
        let namespace = ToolNamespace(name: "remote", description: "Remote tools")
        let tools = [
            ToolDefinition(name: "flat_default", inputSchema: ["type": "object"]),
            ToolDefinition(name: "flat_strict", inputSchema: ["type": "object"], strict: true),
            ToolDefinition(
                name: "nested_default", inputSchema: ["type": "object"], namespace: namespace
            ),
            ToolDefinition(
                name: "nested_strict",
                inputSchema: ["type": "object"],
                strict: true,
                namespace: namespace
            ),
        ]

        let codex = OpenAICodexResponsesRequest.encode(CallOptions(
            model: "gpt-5-codex", prompt: [.user("hello")], tools: tools
        )).body["tools"]
        #expect(codex?[0]?["strict"]?.isNull == true)
        #expect(codex?[1]?["strict"]?.boolValue == true)
        #expect(codex?[2]?["tools"]?[0]?["strict"]?.isNull == true)
        #expect(codex?[2]?["tools"]?[1]?["strict"]?.boolValue == true)

        // With no dialect default, ordinary Responses retains its existing
        // omission for `strict: false` and explicit `true` representation.
        let ordinary = OpenAIResponsesRequest.encode(CallOptions(
            model: "gpt-5", prompt: [.user("hello")], tools: tools
        )).body["tools"]
        #expect(ordinary?[0]?["strict"] == nil)
        #expect(ordinary?[1]?["strict"]?.boolValue == true)
        #expect(ordinary?[2]?["tools"]?[0]?["strict"] == nil)
        #expect(ordinary?[2]?["tools"]?[1]?["strict"]?.boolValue == true)
    }

    @Test("Codex supplies bearer, account, and required headers with normal overrides")
    func buildsCodexHeaders() throws {
        let token = codexJWT(accountId: "account-1")
        let client = AIClient(
            provider: .init(id: "codex", api: nil, speaking: .openAICodex),
            configuration: .init(
                authorization: .oauth(OAuthCredential(accessToken: token, accountId: "retained")),
                extraHeaders: ["OpenAI-Beta": "override", "X-Custom": "yes"]
            )
        )
        let request = try client.makeRequest(
            wire: .openAICodex,
            options: options,
            body: OpenAICodexResponsesRequest.encode(options).body
        )

        #expect(request.url?.absoluteString == "https://chatgpt.com/backend-api/codex/responses")
        #expect(request.value(forHTTPHeaderField: "authorization") == "Bearer \(token)")
        // A fresh JWT claim wins over retained fallback identity.
        #expect(request.value(forHTTPHeaderField: "chatgpt-account-id") == "account-1")
        #expect(request.value(forHTTPHeaderField: "OpenAI-Beta") == "override")
        #expect(request.value(forHTTPHeaderField: "originator") == "aikit-swift")
        #expect(request.value(forHTTPHeaderField: "User-Agent") == "AIKitSwift")
        #expect(request.value(forHTTPHeaderField: "X-Custom") == "yes")
    }

    @Test("a short Codex session identity remains unchanged in body and headers")
    func buildsCodexSessionHeaders() throws {
        let call = CallOptions(
            model: "gpt-5-codex",
            prompt: [.user("hello")],
            providerOptions: ["openai-codex": ["prompt_cache_key": "session-123"]]
        )
        let client = AIClient(
            provider: .init(id: "codex", api: nil, speaking: .openAICodex),
            configuration: .init(authorization: .oauth(OAuthCredential(
                accessToken: codexJWT(accountId: "account")
            )))
        )
        let body = OpenAICodexResponsesRequest.encode(call).body
        let request = try client.makeRequest(wire: .openAICodex, options: call, body: body)

        #expect(body["prompt_cache_key"]?.stringValue == "session-123")
        #expect(request.value(forHTTPHeaderField: "session-id") == "session-123")
        #expect(request.value(forHTTPHeaderField: "x-client-request-id") == "session-123")
    }

    @Test("Codex session identity clamps ASCII at 64 code points")
    func clampsASCIIRequestIdentity() throws {
        let expected = String(repeating: "a", count: 64)
        let key = expected + "z"
        let (body, request) = try makeCodexSessionRequest(promptCacheKey: key)

        #expect(OpenAICodexResponsesRequest.clampPromptCacheKey(expected) == expected)
        #expect(body["prompt_cache_key"]?.stringValue == expected)
        #expect(request.value(forHTTPHeaderField: "session-id") == expected)
        #expect(request.value(forHTTPHeaderField: "x-client-request-id") == expected)
    }

    @Test("Codex session identity clamps Unicode code points without byte truncation")
    func clampsUnicodeRequestIdentity() throws {
        let scalarPrefix = String(repeating: "😀", count: 63)
        let exactBoundary = scalarPrefix + "é"
        // The decomposed accent is a second code point in the same grapheme.
        // JavaScript's `Array.from` limit falls between those two scalars.
        let expected = scalarPrefix + "e"
        let key = scalarPrefix + "e\u{301}z"
        let (body, request) = try makeCodexSessionRequest(promptCacheKey: key)
        let clamped = try #require(body["prompt_cache_key"]?.stringValue)

        #expect(OpenAICodexResponsesRequest.clampPromptCacheKey(exactBoundary) == exactBoundary)
        #expect(clamped == expected)
        #expect(clamped.unicodeScalars.count == 64)
        #expect(clamped.utf8.count > 64)
        #expect(request.value(forHTTPHeaderField: "session-id") == clamped)
        #expect(request.value(forHTTPHeaderField: "x-client-request-id") == clamped)
    }

    @Test("Codex rejects an empty bearer credential")
    func rejectsEmptyCodexCredential() {
        let client = AIClient(
            provider: .init(id: "codex", api: nil, speaking: .openAICodex),
            configuration: .init(authorization: .oauth(OAuthCredential(
                accessToken: "   ", accountId: "account"
            )))
        )

        #expect(throws: AIClientError.self) {
            _ = try client.makeRequest(
                wire: .openAICodex,
                options: options,
                body: OpenAICodexResponsesRequest.encode(options).body
            )
        }
    }

    @Test("ordinary Responses keeps its endpoint, body, and headers")
    func preservesOrdinaryResponses() throws {
        let client = AIClient(
            provider: .init(
                id: "openai",
                api: "https://api.openai.com",
                speaking: .openAIResponses
            ),
            configuration: .init(apiKey: "sk-test")
        )
        let body = client.encode(options, wire: .openAIResponses, model: nil).body
        let request = try client.makeRequest(wire: .openAIResponses, options: options, body: body)

        #expect(request.url?.absoluteString == "https://api.openai.com/v1/responses")
        #expect(body["store"] == nil)
        #expect(body["text"] == nil)
        #expect(body["parallel_tool_calls"] == nil)
        #expect(request.value(forHTTPHeaderField: "chatgpt-account-id") == nil)
        #expect(request.value(forHTTPHeaderField: "OpenAI-Beta") == nil)
        #expect(request.value(forHTTPHeaderField: "authorization") == "Bearer sk-test")
    }

    @Test("Codex maps known and done events while preserving exact payloads")
    func mapsCodexEvents() {
        let known: JSONValue = [
            "type": "response.in_progress",
            "sequence_number": 0,
            "response": ["id": "resp_1", "status": "in_progress"],
        ]
        let done: JSONValue = [
            "type": "response.done",
            "sequence_number": 1,
            "response": [
                "id": "resp_1", "status": "completed",
                "output": [[
                    "type": "message", "id": "msg_1", "role": "assistant",
                    "status": "completed",
                    "content": [["type": "output_text", "text": "hello", "annotations": []]],
                ]],
                "usage": ["input_tokens": 2, "output_tokens": 1],
            ],
        ]
        var wire = OpenAICodexResponsesWire()
        var parts = wire.map(chunk: known)
        parts += wire.map(chunk: done)
        parts += wire.finish()
        let response = AIResponse(parts: parts)

        #expect(response.providerEvents.contains {
            $0.provider == "openai-codex" && $0.type == "response.in_progress" && $0.payload == known
        })
        #expect(response.providerEvents.contains {
            $0.provider == "openai-codex" && $0.type == "response.done" && $0.payload == done
        })
        #expect(response.finishReason?.unified == .stop)
        #expect(
            response.assistantMessage.providerOptions?["openai"]?["outputItems"]?[0]?["id"]?.stringValue
                == "msg_1"
        )
        #expect(response.usage.inputTokens.total == 2)
        #expect(response.usage.outputTokens.total == 1)
    }

    @Test("Codex terminal aliases accept every documented status without becoming failure events")
    func mapsEveryCodexTerminalStatus() {
        let expected: [String: FinishReason.Unified] = [
            "completed": .stop,
            "incomplete": .contentFilter,
            "failed": .error,
            "cancelled": .error,
            "queued": .stop,
            "in_progress": .stop,
        ]

        for (status, expectedReason) in expected {
            let event: JSONValue = [
                "type": "response.done",
                "sequence_number": 7,
                "response": [
                    "id": .string("resp_\(status)"),
                    "status": .string(status),
                    "output": [],
                    "usage": ["input_tokens": 4, "output_tokens": 2],
                ],
            ]
            var wire = OpenAICodexResponsesWire()
            let parts = wire.map(chunk: event) + wire.finish()
            let response = AIResponse(parts: parts)
            let finishMetadata = parts.compactMap { part -> ProviderMetadata? in
                if case .finish(_, _, let metadata) = part { return metadata }
                return nil
            }.first

            #expect(response.finishReason?.unified == expectedReason)
            #expect(response.finishReason?.raw == status)
            #expect(response.errors.isEmpty)
            #expect(response.providerEvents.last?.type == "response.done")
            #expect(response.providerEvents.last?.payload == event)
            #expect(finishMetadata?["openai"]?["response"]?["status"]?.stringValue == status)
            #expect(finishMetadata?["openai"]?["usage"]?["input_tokens"]?.intValue == 4)
        }
    }

    @Test("complete Codex responses use terminal alias semantics")
    func mapsNonStreamingCodexTerminalStatus() {
        let body: JSONValue = [
            "id": "resp_cancelled",
            "status": "cancelled",
            "output": [],
            "usage": ["input_tokens": 3, "output_tokens": 1],
        ]
        let parts = NonStreamingResponse.decode(body, wire: .openAICodex)
        let response = AIResponse(parts: parts)

        #expect(response.finishReason?.unified == .error)
        #expect(response.finishReason?.raw == "cancelled")
        #expect(response.errors.isEmpty)
        #expect(response.providerEvents.last?.provider == "openai-codex")
        #expect(response.providerEvents.last?.type == "response.done")
        #expect(response.providerEvents.last?.payload["response"] == body)
    }

    @Test("an unsupported Codex terminal status is malformed and retains its payload")
    func rejectsUnsupportedCodexTerminalStatus() {
        let event: JSONValue = [
            "type": "response.done",
            "sequence_number": 0,
            "response": ["id": "resp_1", "status": "future", "output": []],
        ]
        var wire = OpenAICodexResponsesWire()
        let response = AIResponse(parts: wire.map(chunk: event))

        #expect(response.providerEvents.last?.payload == event)
        #expect(response.errors.first?.type == "malformed_event")
        #expect(response.errors.first?.message.contains("unsupported Codex value") == true)
        #expect(response.errors.first?.raw == event)
    }

    @Test("only an actual Codex response.failed event emits the provider failure")
    func distinguishesActualCodexFailureEvent() {
        let event: JSONValue = [
            "type": "response.failed",
            "sequence_number": 0,
            "response": [
                "id": "resp_1", "status": "failed", "output": [],
                "error": ["code": "bad_request", "message": "failed for real"],
            ],
        ]
        var wire = OpenAICodexResponsesWire()
        let response = AIResponse(parts: wire.map(chunk: event))

        #expect(response.providerEvents.last?.type == "response.failed")
        #expect(response.providerEvents.last?.payload == event)
        #expect(response.errors.first?.type == "bad_request")
        #expect(response.errors.first?.message == "failed for real")
        #expect(response.errors.first?.raw == event)
    }

    @Test("Codex malformed known events retain the original payload")
    func rejectsMalformedCodexEvent() {
        let malformed: JSONValue = [
            "type": "response.output_text.delta",
            "sequence_number": 0,
            "item_id": "msg_1", "output_index": 0, "content_index": 0,
            "delta": 7,
        ]
        var wire = OpenAICodexResponsesWire()
        let parts = wire.map(chunk: malformed)

        #expect(parts.contains {
            if case .providerEvent(let event) = $0 {
                return event.provider == "openai-codex" && event.payload == malformed
            }
            return false
        })
        #expect(parts.contains {
            if case .error(let error) = $0 {
                return error.type == "malformed_event"
                    && error.message.contains("OpenAI Codex Responses")
                    && error.raw == malformed
            }
            return false
        })
    }

    @Test("Codex function argument completion requires name and retains malformed payload")
    func requiresCodexFunctionArgumentName() {
        let malformed: JSONValue = [
            "type": "response.function_call_arguments.done",
            "sequence_number": 0,
            "item_id": "fc_1",
            "output_index": 0,
            "arguments": "{}",
        ]
        var wire = OpenAICodexResponsesWire()
        let response = AIResponse(parts: wire.map(chunk: malformed))

        #expect(response.providerEvents.last?.payload == malformed)
        #expect(response.errors.first?.type == "malformed_event")
        #expect(response.errors.first?.message.contains("name") == true)
        #expect(response.errors.first?.raw == malformed)
    }

    @Test("Codex unknown valid events remain raw")
    func preservesUnknownCodexEvent() {
        let unknown: JSONValue = ["type": "response.future", "future": ["enabled": true]]
        var wire = OpenAICodexResponsesWire()
        let parts = wire.map(chunk: unknown)

        #expect(parts.contains {
            if case .raw(let payload) = $0 { return payload == unknown }
            return false
        })
    }

    @Test("a Codex 401 refreshes and replays exactly once")
    func refreshesAndReplaysOnce() async throws {
        let router = CodexURLProtocol.router
        await router.configure([
            .init(status: 401, body: Data("unauthorized".utf8)),
            .init(status: 200, body: codexDoneSSE),
        ])
        let store = MemoryOAuthCredentialStore()
        let old = OAuthCredential(
            accessToken: codexJWT(accountId: "old-account"),
            refreshToken: "refresh",
            expiresAt: Date.distantFuture,
            accountId: "old-account"
        )
        let refreshed = OAuthCredential(
            accessToken: codexJWT(accountId: "new-account"),
            refreshToken: "refresh-2",
            expiresAt: Date.distantFuture,
            accountId: "new-account"
        )
        let probe = RefreshProbe(result: refreshed)
        let manager = OAuthCredentialManager(
            key: "codex",
            initialCredential: old,
            store: store
        ) { credential in
            try await probe.refresh(credential)
        }
        let client = codexClient(
            session: codexSession(),
            credentialProvider: manager
        )

        let response = try await client.stream(options).collect()
        let requests = await router.recordedRequests()

        #expect(response.finishReason?.unified == .stop)
        #expect(requests.count == 2)
        #expect(requests[0].value(forHTTPHeaderField: "chatgpt-account-id") == "old-account")
        #expect(requests[1].value(forHTTPHeaderField: "chatgpt-account-id") == "new-account")
        #expect(requests[0].httpBody == requests[1].httpBody)
        #expect(await probe.count == 1)
        #expect(await store.savedCredentials == [old, refreshed])
    }

    @Test("Codex generate collects its supported SSE transport")
    func generateCollectsCodexSSE() async throws {
        let router = CodexURLProtocol.router
        await router.configure([.init(status: 200, body: codexTextDoneSSE)])
        let client = AIClient(
            provider: .init(id: "codex", api: nil, speaking: .openAICodex),
            configuration: .init(authorization: .oauth(OAuthCredential(
                accessToken: codexJWT(accountId: "account")
            ))),
            session: codexSession()
        )

        let response = try await client.generate(options)
        let requests = await router.recordedRequests()
        let request = try #require(requests.first)
        let body = try JSONValue.decode(from: try #require(request.httpBody))

        #expect(response.text == "hello")
        #expect(response.finishReason?.unified == .stop)
        #expect(response.errors.isEmpty)
        #expect(requests.count == 1)
        #expect(body["stream"]?.boolValue == true)
        #expect(request.value(forHTTPHeaderField: "accept") == "text/event-stream")
    }

    @Test("Codex generate refreshes and replays its SSE request exactly once")
    func refreshesAndReplaysGenerateOnce() async throws {
        let router = CodexURLProtocol.router
        await router.configure([
            .init(status: 401, body: Data("unauthorized".utf8)),
            .init(status: 200, body: codexDoneSSE),
        ])
        let old = OAuthCredential(
            accessToken: codexJWT(accountId: "old-account"),
            refreshToken: "refresh",
            expiresAt: Date.distantFuture
        )
        let refreshed = OAuthCredential(
            accessToken: codexJWT(accountId: "new-account"),
            refreshToken: "refresh-2",
            expiresAt: Date.distantFuture
        )
        let probe = RefreshProbe(result: refreshed)
        let manager = OAuthCredentialManager(
            key: "codex",
            initialCredential: old,
            store: MemoryOAuthCredentialStore()
        ) { credential in
            try await probe.refresh(credential)
        }
        let client = codexClient(session: codexSession(), credentialProvider: manager)

        let response = try await client.generate(options)
        let requests = await router.recordedRequests()

        #expect(response.finishReason?.unified == .stop)
        #expect(requests.count == 2)
        #expect(requests[0].httpBody == requests[1].httpBody)
        #expect(requests.allSatisfy {
            $0.value(forHTTPHeaderField: "accept") == "text/event-stream"
        })
        let bodies = try requests.map { request in
            try JSONValue.decode(from: try #require(request.httpBody))
        }
        #expect(bodies.allSatisfy { $0["stream"]?.boolValue == true })
        #expect(requests[0].value(forHTTPHeaderField: "chatgpt-account-id") == "old-account")
        #expect(requests[1].value(forHTTPHeaderField: "chatgpt-account-id") == "new-account")
        #expect(await probe.count == 1)
    }

    @Test("a second 401 is surfaced without another refresh")
    func doesNotLoopAuthenticationReplay() async {
        let router = CodexURLProtocol.router
        await router.configure([
            .init(status: 401, body: Data("first".utf8)),
            .init(status: 401, body: Data("second".utf8)),
        ])
        let store = MemoryOAuthCredentialStore()
        let old = OAuthCredential(
            accessToken: codexJWT(accountId: "old"),
            refreshToken: "refresh",
            expiresAt: Date.distantFuture
        )
        let probe = RefreshProbe(result: OAuthCredential(
            accessToken: codexJWT(accountId: "new"),
            refreshToken: "new-refresh",
            expiresAt: Date.distantFuture
        ))
        let manager = OAuthCredentialManager(
            key: "codex", initialCredential: old, store: store
        ) { credential in
            try await probe.refresh(credential)
        }
        let client = codexClient(session: codexSession(), credentialProvider: manager)

        do {
            _ = try await client.stream(options).collect()
            Issue.record("Expected the second 401 to be surfaced")
        } catch let error as AIClientError {
            guard case .http(let status, let body) = error.kind else {
                Issue.record("Expected an HTTP error")
                return
            }
            #expect(status == 401)
            #expect(body == "second")
        } catch {
            Issue.record("Unexpected error: \(error)")
        }

        #expect(await router.recordedRequests().count == 2)
        #expect(await probe.count == 1)
    }

    @Test("unrelated failures are not refreshed or retried")
    func doesNotRetryUnrelatedFailure() async {
        let router = CodexURLProtocol.router
        await router.configure([.init(status: 500, body: Data("server".utf8))])
        let store = MemoryOAuthCredentialStore()
        let credential = OAuthCredential(
            accessToken: codexJWT(accountId: "account"),
            refreshToken: "refresh",
            expiresAt: Date.distantFuture
        )
        let probe = RefreshProbe(result: credential)
        let manager = OAuthCredentialManager(
            key: "codex", initialCredential: credential, store: store
        ) { value in
            try await probe.refresh(value)
        }
        let client = codexClient(session: codexSession(), credentialProvider: manager)

        do {
            _ = try await client.stream(options).collect()
            Issue.record("Expected HTTP 500")
        } catch let error as AIClientError {
            guard case .http(let status, _) = error.kind else {
                Issue.record("Expected an HTTP error")
                return
            }
            #expect(status == 500)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }

        #expect(await router.recordedRequests().count == 1)
        #expect(await probe.count == 0)
    }

    @Test("cancelling a Codex stream cancels its URLSession request")
    func preservesCancellation() async throws {
        let router = CodexURLProtocol.router
        await router.configure([.init(status: 200, body: Data(), waitsForCancellation: true)])
        let credential = OAuthCredential(
            accessToken: codexJWT(accountId: "account"),
            expiresAt: Date.distantFuture
        )
        let client = AIClient(
            provider: .init(id: "codex", api: nil, speaking: .openAICodex),
            configuration: .init(authorization: .oauth(credential)),
            session: codexSession()
        )

        let task = Task { try await client.stream(options).collect() }
        await router.waitForRequests(1)
        task.cancel()

        do {
            _ = try await task.value
        } catch {
            let cancelled = error is CancellationError
                || (error as? URLError)?.code == .cancelled
            #expect(cancelled)
        }
        await router.waitForCancellations(1)
        #expect(await router.cancellationCount == 1)
        #expect(await router.recordedRequests().count == 1)
    }

    @Test("a Codex terminal event cancels an HTTP body that remains open")
    func terminalEventCancelsTransport() async throws {
        let router = CodexURLProtocol.router
        await router.configure([.init(
            status: 200,
            body: codexDoneSSE,
            waitsAfterBodyForCancellation: true
        )])
        let client = AIClient(
            provider: .init(id: "codex", api: nil, speaking: .openAICodex),
            configuration: .init(authorization: .oauth(OAuthCredential(
                accessToken: codexJWT(accountId: "account")
            ))),
            session: codexSession()
        )

        let response = try await client.stream(options).collect()
        await router.waitForCancellations(1)

        #expect(response.finishReason?.unified == .stop)
        #expect(response.errors.isEmpty)
        #expect(await router.cancellationCount == 1)
    }

    @Test("Codex EOF without a terminal response is a normalized stream error")
    func rejectsCodexEOFWithoutTerminalEvent() async throws {
        let router = CodexURLProtocol.router
        let incomplete = Data(
            "data: {\"type\":\"response.in_progress\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"status\":\"in_progress\"}}\n\n".utf8
        )
        await router.configure([.init(status: 200, body: incomplete)])
        let client = AIClient(
            provider: .init(id: "codex", api: nil, speaking: .openAICodex),
            configuration: .init(authorization: .oauth(OAuthCredential(
                accessToken: codexJWT(accountId: "account")
            ))),
            session: codexSession()
        )

        let response = try await client.stream(options).collect()

        #expect(response.errors.first?.type == "incomplete_stream")
        #expect(response.errors.first?.message ==
            "OpenAI Codex Responses stream ended before a terminal response event")
        #expect(response.providerEvents.last?.type == "response.in_progress")
    }

    @Test("Codex live model listing is explicitly disabled without network I/O")
    func disablesCodexLiveModelListing() async {
        let router = CodexURLProtocol.router
        await router.configure([])
        let client = AIClient(
            provider: .init(id: "codex", api: nil, speaking: .openAICodex),
            configuration: .init(authorization: .oauth(OAuthCredential(
                accessToken: codexJWT(accountId: "account")
            ))),
            session: codexSession()
        )

        do {
            _ = try await client.models()
            Issue.record("Expected Codex live model listing to be disabled")
        } catch let error as AIClientError {
            guard case .unsupportedModelListing(let wire) = error.kind else {
                Issue.record("Expected unsupportedModelListing, got \(error)")
                return
            }
            #expect(wire == WireProtocol.openAICodex.rawValue)
        } catch {
            Issue.record("Unexpected error: \(error)")
        }
        #expect(await router.recordedRequests().isEmpty)
    }

    @Test("Codex PKCE binds the registered callback, validates state, exchanges, and persists")
    func runsCompleteCodexPKCEFlow() async throws {
        let router = CodexURLProtocol.router
        let token = codexJWT(accountId: "pkce-account")
        await router.configure([.init(
            status: 200,
            body: jsonData([
                "access_token": .string(token),
                "refresh_token": "refresh-1",
                "expires_in": 3600,
            ]),
            headers: ["content-type": "application/json"]
        )])
        let configuration = OpenAICodexOAuthConfiguration(
            authBaseURL: URL(string: "https://auth.example.com")!,
            browserRedirectURI: "http://localhost:1455/auth/callback",
            originator: "tests"
        )
        let store = MemoryOAuthCredentialStore()
        let browser = AuthorizationURLRecorder()
        let client = OpenAICodexOAuthClient(
            configuration: configuration,
            session: codexSession(),
            credentialStore: store,
            now: { Date(timeIntervalSince1970: 1_000) }
        )
        let credential = try await client.authorizePKCE(timeout: 10) { url in
            browser.record(url)
            let query = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
            let state = query.first { $0.name == "state" }?.value ?? ""
            let redirect = query.first { $0.name == "redirect_uri" }?.value ?? ""
            Task {
                let callback = URL(string: "\(redirect)?code=code-1&state=\(state)")!
                _ = try? await URLSession.shared.data(from: callback)
            }
        }

        let url = try #require(browser.value)
        let query = URLComponents(url: url, resolvingAgainstBaseURL: false)?.queryItems ?? []
        func value(_ name: String) -> String? { query.first { $0.name == name }?.value }

        #expect(value("client_id") == OpenAICodexOAuthConfiguration.clientId)
        #expect(value("scope") == "openid profile email offline_access")
        #expect(value("redirect_uri") == "http://localhost:1455/auth/callback")
        #expect(value("code_challenge")?.isEmpty == false)
        #expect(value("state")?.isEmpty == false)
        #expect(value("id_token_add_organizations") == "true")
        #expect(value("codex_cli_simplified_flow") == "true")
        #expect(value("originator") == "tests")

        let request = try #require(await router.recordedRequests().first)
        let body = String(decoding: try #require(request.httpBody), as: UTF8.self)
        var form = URLComponents()
        form.percentEncodedQuery = body
        let formItems = form.queryItems ?? []
        func formValue(_ name: String) -> String? { formItems.first { $0.name == name }?.value }

        #expect(request.url?.absoluteString == "https://auth.example.com/oauth/token")
        #expect(formValue("grant_type") == "authorization_code")
        #expect(formValue("code") == "code-1")
        #expect(formValue("redirect_uri") == "http://localhost:1455/auth/callback")
        let verifier = try #require(formValue("code_verifier"))
        #expect(PKCE.challenge(for: verifier) == value("code_challenge"))
        #expect(credential.accountId == "pkce-account")
        #expect(credential.refreshToken == "refresh-1")
        #expect(credential.expiresAt == Date(timeIntervalSince1970: 4_600))
        #expect(await store.savedCredentials == [credential])
        let reloaded = client.credentialManager(refreshLeeway: 0)
        #expect(try await reloaded.credential() == credential)
    }

    @Test("Codex PKCE rejects a listener that cannot receive its registered redirect")
    func rejectsMismatchedCodexPKCEListener() async throws {
        let client = OpenAICodexOAuthClient(
            credentialStore: MemoryOAuthCredentialStore()
        )
        let listener = try LoopbackRedirectListener()

        await #expect(throws: OpenAICodexOAuthError.self) {
            _ = try await client.authorizePKCE(listener: listener) { _ in }
        }
    }

    @Test("Codex refresh retains account and refresh identity when omitted")
    func retainsCredentialIdentityAcrossRefresh() async throws {
        let router = CodexURLProtocol.router
        await router.configure([.init(
            status: 200,
            body: jsonData(["access_token": "opaque-new-token", "expires_in": 3600]),
            headers: ["content-type": "application/json"]
        )])
        let old = OAuthCredential(
            accessToken: "opaque-old-token",
            refreshToken: "refresh-old",
            expiresAt: Date(timeIntervalSince1970: 900),
            accountId: "retained-account"
        )
        let oauth = OpenAICodexOAuthClient(
            configuration: .init(authBaseURL: URL(string: "https://auth.example.com")!),
            session: codexSession(),
            now: { Date(timeIntervalSince1970: 1_000) }
        )

        let refreshed = try await oauth.refresh(old)
        let tokenRequest = try #require(await router.recordedRequests().first)
        let body = String(decoding: try #require(tokenRequest.httpBody), as: UTF8.self)

        #expect(refreshed.accessToken == "opaque-new-token")
        #expect(refreshed.refreshToken == "refresh-old")
        #expect(refreshed.accountId == "retained-account")
        #expect(refreshed.expiresAt == Date(timeIntervalSince1970: 4_600))
        #expect(body.contains("grant_type=refresh_token"))
        #expect(body.contains("refresh_token=refresh-old"))

        let request = try AIClient(
            provider: .init(id: "codex", api: nil, speaking: .openAICodex),
            configuration: .init(authorization: .oauth(refreshed))
        ).makeRequest(
            wire: .openAICodex,
            options: options,
            body: OpenAICodexResponsesRequest.encode(options).body
        )
        #expect(request.value(forHTTPHeaderField: "chatgpt-account-id") == "retained-account")
    }

    @Test("Codex refresh derives fallback account identity from the prior access JWT")
    func derivesRefreshFallbackAccountFromPriorJWT() async throws {
        let router = CodexURLProtocol.router
        await router.configure([.init(
            status: 200,
            body: jsonData(["access_token": "opaque-new-token", "expires_in": 3600]),
            headers: ["content-type": "application/json"]
        )])
        let old = OAuthCredential(
            accessToken: codexJWT(accountId: "jwt-fallback"),
            refreshToken: "refresh-old",
            expiresAt: Date(timeIntervalSince1970: 900),
            accountId: nil
        )
        let oauth = OpenAICodexOAuthClient(
            configuration: .init(authBaseURL: URL(string: "https://auth.example.com")!),
            session: codexSession(),
            now: { Date(timeIntervalSince1970: 1_000) }
        )

        let refreshed = try await oauth.refresh(old)

        #expect(refreshed.accountId == "jwt-fallback")
        #expect(refreshed.refreshToken == "refresh-old")
    }

    @Test("Codex device flow polls, exchanges, and reports the user code")
    func runsCodexDeviceCodeFlow() async throws {
        let router = CodexURLProtocol.router
        let token = codexJWT(accountId: "device-account")
        await router.configure([
            .init(
                status: 200,
                body: jsonData([
                    "device_auth_id": "device-auth-1",
                    "user_code": "ABCD-1234",
                    "interval": "5",
                ]),
                headers: ["content-type": "application/json"]
            ),
            .init(
                status: 403,
                body: jsonData(["error": ["code": "deviceauth_authorization_pending"]]),
                headers: ["content-type": "application/json"]
            ),
            .init(
                status: 200,
                body: jsonData([
                    "authorization_code": "device-code",
                    "code_verifier": "device-verifier",
                ]),
                headers: ["content-type": "application/json"]
            ),
            .init(
                status: 200,
                body: jsonData([
                    "access_token": .string(token),
                    "refresh_token": "device-refresh",
                    "expires_in": 3600,
                ]),
                headers: ["content-type": "application/json"]
            ),
        ])
        let sleeps = SleepRecorder()
        let codeRecorder = DeviceCodeRecorder()
        let store = MemoryOAuthCredentialStore()
        let client = OpenAICodexOAuthClient(
            configuration: .init(authBaseURL: URL(string: "https://auth.example.com")!),
            session: codexSession(),
            credentialStore: store,
            now: Date.init,
            sleep: { seconds in await sleeps.record(seconds) }
        )

        let credential = try await client.authorizeDeviceCode { codeRecorder.record($0) }
        let requests = await router.recordedRequests()
        let code = codeRecorder.value

        #expect(code?.userCode == "ABCD-1234")
        #expect(code?.verificationURL.absoluteString == "https://auth.example.com/codex/device")
        #expect(code?.interval == 5)
        #expect(await sleeps.values == [5])
        #expect(requests.map(\.url?.path) == [
            "/api/accounts/deviceauth/usercode",
            "/api/accounts/deviceauth/token",
            "/api/accounts/deviceauth/token",
            "/oauth/token",
        ])
        let exchange = String(decoding: try #require(requests.last?.httpBody), as: UTF8.self)
        #expect(exchange.contains("code=device-code"))
        #expect(exchange.contains("code_verifier=device-verifier"))
        #expect(exchange.contains("redirect_uri=https://auth.example.com/deviceauth/callback"))
        #expect(credential.accountId == "device-account")
        #expect(await store.savedCredentials == [credential])
    }

    @Test("persisted expired credentials load, refresh, save, and reload")
    func persistsAndRefreshesExpiredCredential() async throws {
        let now = Date(timeIntervalSince1970: 10_000)
        let expired = OAuthCredential(
            accessToken: codexJWT(accountId: "old"),
            refreshToken: "refresh-old",
            expiresAt: now.addingTimeInterval(-1),
            accountId: "old"
        )
        let refreshed = OAuthCredential(
            accessToken: codexJWT(accountId: "new"),
            refreshToken: "refresh-new",
            expiresAt: now.addingTimeInterval(3_600),
            accountId: "new"
        )
        let store = MemoryOAuthCredentialStore(expired)
        let probe = RefreshProbe(result: refreshed)
        let manager = OAuthCredentialManager(
            key: "codex",
            store: store,
            now: { now }
        ) { credential in
            try await probe.refresh(credential)
        }

        let first = try await manager.credential()
        #expect(first == refreshed)
        #expect(await store.loadCount == 1)
        #expect(await store.savedCredentials == [refreshed])
        #expect(await probe.count == 1)

        let reloadProbe = RefreshProbe(result: refreshed)
        let reloaded = OAuthCredentialManager(
            key: "codex",
            store: store,
            now: { now }
        ) { credential in
            try await reloadProbe.refresh(credential)
        }
        #expect(try await reloaded.credential() == refreshed)
        #expect(await store.loadCount == 2)
        #expect(await reloadProbe.count == 0)
    }

    @Test("simultaneous expired credential requests share one refresh")
    func coalescesConcurrentRefresh() async throws {
        let expired = OAuthCredential(
            accessToken: codexJWT(accountId: "old"),
            refreshToken: "refresh",
            expiresAt: Date.distantPast
        )
        let refreshed = OAuthCredential(
            accessToken: codexJWT(accountId: "new"),
            refreshToken: "refresh-2",
            expiresAt: Date.distantFuture,
            accountId: "new"
        )
        let store = MemoryOAuthCredentialStore()
        let gate = RefreshGate(result: refreshed)
        let manager = OAuthCredentialManager(
            key: "codex", initialCredential: expired, store: store
        ) { credential in
            await gate.refresh(credential)
        }

        let tasks = (0..<12).map { _ in Task { try await manager.credential() } }
        await gate.waitUntilEntered()
        #expect(await gate.count == 1)
        await gate.release()
        let values = try await tasks.asyncMap { try await $0.value }

        #expect(values.allSatisfy { $0 == refreshed })
        #expect(await gate.count == 1)
        #expect(await store.savedCredentials == [expired, refreshed])
    }

    @Test("a cancelled refresh waiter detaches while the shared refresh completes")
    func cancellationDoesNotCancelSharedRefresh() async throws {
        let expired = OAuthCredential(
            accessToken: codexJWT(accountId: "old"),
            refreshToken: "refresh",
            expiresAt: .distantPast
        )
        let refreshed = OAuthCredential(
            accessToken: codexJWT(accountId: "new"),
            refreshToken: "refresh-2",
            expiresAt: .distantFuture,
            accountId: "new"
        )
        let store = MemoryOAuthCredentialStore()
        let gate = RefreshGate(result: refreshed)
        let manager = OAuthCredentialManager(
            key: "codex", initialCredential: expired, store: store
        ) { credential in
            await gate.refresh(credential)
        }

        let cancelled = Task { try await manager.credential() }
        let remaining = Task { try await manager.credential() }
        await gate.waitUntilEntered()
        cancelled.cancel()

        do {
            _ = try await cancelled.value
            Issue.record("Expected the cancelled waiter to detach")
        } catch {
            #expect(error is CancellationError)
        }
        #expect(await gate.count == 1)

        await gate.release()
        #expect(try await remaining.value == refreshed)
        #expect(await gate.count == 1)
        #expect(await store.savedCredentials == [expired, refreshed])
    }

    @Test("install waits for an in-flight refresh and remains the final credential")
    func installDoesNotRaceInFlightRefresh() async throws {
        let expired = OAuthCredential(
            accessToken: codexJWT(accountId: "expired"),
            refreshToken: "refresh-expired",
            expiresAt: .distantPast
        )
        let refreshed = OAuthCredential(
            accessToken: codexJWT(accountId: "refreshed"),
            refreshToken: "refresh-next",
            expiresAt: .distantFuture
        )
        let installed = OAuthCredential(
            accessToken: codexJWT(accountId: "installed"),
            refreshToken: "refresh-installed",
            expiresAt: .distantFuture
        )
        let store = MemoryOAuthCredentialStore(expired)
        let refresh = RefreshGate(result: refreshed)
        let manager = OAuthCredentialManager(key: "codex", store: store) { credential in
            await refresh.refresh(credential)
        }

        let request = Task { try await manager.credential() }
        await refresh.waitUntilEntered()
        let invoked = StartGate()
        let installation = Task {
            await invoked.release()
            try await manager.install(installed)
        }
        await invoked.wait()
        for _ in 0..<10 { await Task.yield() }
        #expect(await store.savedCredentials.isEmpty)

        await refresh.release()
        #expect(try await request.value == refreshed)
        try await installation.value
        #expect(try await manager.credential() == installed)
        #expect(await store.savedCredentials == [refreshed, installed])
    }

    @Test("delete waits for an in-flight refresh and remains deleted")
    func deleteDoesNotRaceInFlightRefresh() async throws {
        let expired = OAuthCredential(
            accessToken: codexJWT(accountId: "expired"),
            refreshToken: "refresh-expired",
            expiresAt: .distantPast
        )
        let refreshed = OAuthCredential(
            accessToken: codexJWT(accountId: "refreshed"),
            refreshToken: "refresh-next",
            expiresAt: .distantFuture
        )
        let store = MemoryOAuthCredentialStore(expired)
        let refresh = RefreshGate(result: refreshed)
        let manager = OAuthCredentialManager(key: "codex", store: store) { credential in
            await refresh.refresh(credential)
        }

        let request = Task { try await manager.credential() }
        await refresh.waitUntilEntered()
        let invoked = StartGate()
        let deletion = Task {
            await invoked.release()
            try await manager.deleteCredential()
        }
        await invoked.wait()
        for _ in 0..<10 { await Task.yield() }
        #expect(await store.savedCredentials.isEmpty)

        await refresh.release()
        #expect(try await request.value == refreshed)
        try await deletion.value
        await #expect(throws: OAuthCredentialManagerError.self) {
            _ = try await manager.credential()
        }
        #expect(await store.savedCredentials == [refreshed, nil])
    }

    @Test("a pre-cancelled caller cannot initiate credential persistence or refresh")
    func preCancelledCallerDoesNotStartRefresh() async {
        let expired = OAuthCredential(
            accessToken: codexJWT(accountId: "old"),
            refreshToken: "refresh",
            expiresAt: .distantPast
        )
        let store = MemoryOAuthCredentialStore()
        let probe = RefreshProbe(result: expired)
        let start = StartGate()
        let manager = OAuthCredentialManager(
            key: "codex", initialCredential: expired, store: store
        ) { credential in
            try await probe.refresh(credential)
        }

        let caller = Task {
            await start.wait()
            return try await manager.credential()
        }
        caller.cancel()
        await start.release()

        do {
            _ = try await caller.value
            Issue.record("Expected cancellation")
        } catch {
            #expect(error is CancellationError)
        }
        #expect(await probe.count == 0)
        #expect(await store.savedCredentials.isEmpty)
    }

    @Test("authorized credentials install, reload, expire, refresh, and delete")
    func installsReloadsAndDeletesCredential() async throws {
        let now = Date(timeIntervalSince1970: 10_000)
        let authorized = OAuthCredential(
            accessToken: codexJWT(accountId: "authorized"),
            refreshToken: "refresh-authorized",
            expiresAt: now.addingTimeInterval(60),
            accountId: "authorized"
        )
        let refreshed = OAuthCredential(
            accessToken: codexJWT(accountId: "refreshed"),
            refreshToken: "refresh-next",
            expiresAt: now.addingTimeInterval(3_600),
            accountId: "refreshed"
        )
        let store = MemoryOAuthCredentialStore()
        let installManager = OAuthCredentialManager(
            key: "codex", store: store, refreshLeeway: 0, now: { now }
        ) { _ in refreshed }
        try await installManager.install(authorized)

        let probe = RefreshProbe(result: refreshed)
        let reloaded = OAuthCredentialManager(
            key: "codex", store: store, refreshLeeway: 300, now: { now }
        ) { credential in
            try await probe.refresh(credential)
        }
        #expect(try await reloaded.credential() == refreshed)
        #expect(await probe.count == 1)

        try await reloaded.deleteCredential()
        let afterDeletion = OAuthCredentialManager(
            key: "codex", store: store, now: { now }
        ) { credential in credential }
        await #expect(throws: OAuthCredentialManagerError.self) {
            _ = try await afterDeletion.credential()
        }
        #expect(await store.savedCredentials == [authorized, refreshed, nil])
    }

    @Test("device polling bounds pending sleep by the authorization deadline")
    func pendingDeviceFlowStopsAtDeadline() async throws {
        let router = CodexURLProtocol.router
        await router.configure([
            .init(status: 200, body: jsonData([
                "device_auth_id": "device-1", "user_code": "CODE", "interval": 5,
            ]), headers: ["content-type": "application/json"]),
            .init(status: 403, body: jsonData([
                "error": ["code": "deviceauth_authorization_pending"],
            ]), headers: ["content-type": "application/json"]),
        ])
        let clock = LockedTestClock(Date(timeIntervalSince1970: 1_000))
        let client = OpenAICodexOAuthClient(
            configuration: .init(authBaseURL: URL(string: "https://auth.example.com")!),
            session: codexSession(),
            credentialStore: MemoryOAuthCredentialStore(),
            deviceAuthorizationTimeout: 3,
            now: { clock.now() },
            sleep: { clock.advance(by: $0) }
        )

        do {
            _ = try await client.authorizeDeviceCode()
            Issue.record("Expected the pending device flow to time out")
        } catch let error as OpenAICodexOAuthError {
            guard case .deviceFlowTimedOut = error else {
                Issue.record("Expected deviceFlowTimedOut, got \(error)")
                return
            }
        }
        #expect(clock.sleeps == [3])
        #expect(await router.recordedRequests().count == 2)
    }

    @Test("device polling bounds slow-down sleep by the authorization deadline")
    func slowDownDeviceFlowStopsAtDeadline() async throws {
        let router = CodexURLProtocol.router
        await router.configure([
            .init(status: 200, body: jsonData([
                "device_auth_id": "device-1", "user_code": "CODE", "interval": 1,
            ]), headers: ["content-type": "application/json"]),
            .init(status: 400, body: jsonData([
                "error": ["code": "slow_down"],
            ]), headers: ["content-type": "application/json"]),
        ])
        let clock = LockedTestClock(Date(timeIntervalSince1970: 1_000))
        let client = OpenAICodexOAuthClient(
            configuration: .init(authBaseURL: URL(string: "https://auth.example.com")!),
            session: codexSession(),
            credentialStore: MemoryOAuthCredentialStore(),
            deviceAuthorizationTimeout: 3,
            now: { clock.now() },
            sleep: { clock.advance(by: $0) }
        )

        do {
            _ = try await client.authorizeDeviceCode()
            Issue.record("Expected the slowed device flow to time out")
        } catch let error as OpenAICodexOAuthError {
            guard case .deviceFlowTimedOut = error else {
                Issue.record("Expected deviceFlowTimedOut, got \(error)")
                return
            }
        }
        #expect(clock.sleeps == [3])
        #expect(await router.recordedRequests().count == 2)
    }

    @Test("Codex token responses require a valid positive lifetime or usable JWT expiry")
    func validatesCodexTokenLifetime() async throws {
        let router = CodexURLProtocol.router
        let now = Date(timeIntervalSince1970: 1_000)
        let client = OpenAICodexOAuthClient(
            configuration: .init(authBaseURL: URL(string: "https://auth.example.com")!),
            session: codexSession(),
            credentialStore: MemoryOAuthCredentialStore(),
            now: { now }
        )

        let invalidResponses: [JSONValue] = [
            [
                "access_token": .string(codexJWT(accountId: "missing")),
                "refresh_token": "refresh",
            ],
            [
                "access_token": .string(codexJWT(accountId: "invalid")),
                "refresh_token": "refresh",
                "expires_in": "3600",
            ],
            [
                "access_token": .string(codexJWT(accountId: "negative")),
                "refresh_token": "refresh",
                "expires_in": -1,
            ],
            [
                "access_token": .string(codexJWT(
                    accountId: "expired-jwt",
                    expiresAt: now.addingTimeInterval(-1)
                )),
                "refresh_token": "refresh",
            ],
        ]
        for response in invalidResponses {
            await router.configure([.init(
                status: 200,
                body: jsonData(response),
                headers: ["content-type": "application/json"]
            )])
            do {
                _ = try await client.exchange(
                    authorizationCode: "code",
                    pkce: PKCE(verifier: "verifier")
                )
                Issue.record("Expected malformed token lifetime for \(response)")
            } catch let error as OpenAICodexOAuthError {
                guard case .malformedResponse = error else {
                    Issue.record("Expected malformedResponse, got \(error)")
                    continue
                }
            }
        }

        let jwtExpiry = Date(timeIntervalSince1970: 5_000)
        await router.configure([.init(
            status: 200,
            body: jsonData([
                "access_token": .string(codexJWT(accountId: "jwt", expiresAt: jwtExpiry)),
                "refresh_token": "refresh",
            ]),
            headers: ["content-type": "application/json"]
        )])
        let credential = try await client.exchange(
            authorizationCode: "code",
            pkce: PKCE(verifier: "verifier")
        )
        #expect(credential.expiresAt == jwtExpiry)
    }

    @Test("Codex rejects empty required token and device authorization fields")
    func rejectsEmptyOAuthAndDeviceFields() async throws {
        let router = CodexURLProtocol.router
        let client = OpenAICodexOAuthClient(
            configuration: .init(authBaseURL: URL(string: "https://auth.example.com")!),
            session: codexSession(),
            credentialStore: MemoryOAuthCredentialStore()
        )

        for body: JSONValue in [
            ["device_auth_id": "", "user_code": "CODE", "interval": 1],
            ["device_auth_id": "device", "user_code": "   ", "interval": 1],
        ] {
            await router.configure([.init(
                status: 200, body: jsonData(body), headers: ["content-type": "application/json"]
            )])
            await #expect(throws: OpenAICodexOAuthError.self) {
                _ = try await client.startDeviceAuthorization()
            }
        }

        for poll: JSONValue in [
            ["authorization_code": "", "code_verifier": "verifier"],
            ["authorization_code": "code", "code_verifier": "   "],
        ] {
            await router.configure([
                .init(status: 200, body: jsonData([
                    "device_auth_id": "device", "user_code": "CODE", "interval": 1,
                ]), headers: ["content-type": "application/json"]),
                .init(status: 200, body: jsonData(poll), headers: ["content-type": "application/json"]),
            ])
            await #expect(throws: OpenAICodexOAuthError.self) {
                _ = try await client.authorizeDeviceCode()
            }
        }

        for token: JSONValue in [
            ["access_token": "", "refresh_token": "refresh", "expires_in": 3600],
            ["access_token": .string(codexJWT(accountId: "account")), "refresh_token": " ", "expires_in": 3600],
        ] {
            await router.configure([.init(
                status: 200, body: jsonData(token), headers: ["content-type": "application/json"]
            )])
            await #expect(throws: OpenAICodexOAuthError.self) {
                _ = try await client.exchange(
                    authorizationCode: "code", pkce: PKCE(verifier: "verifier")
                )
            }
        }

        await router.configure([])
        await #expect(throws: OpenAICodexOAuthError.self) {
            _ = try await client.exchange(
                authorizationCode: " ", pkce: PKCE(verifier: "verifier")
            )
        }
        await #expect(throws: OpenAICodexOAuthError.self) {
            _ = try await client.exchange(
                authorizationCode: "code", pkce: PKCE(verifier: " ")
            )
        }
        await #expect(throws: OAuthCredentialManagerError.self) {
            _ = try await client.refresh(OAuthCredential(
                accessToken: codexJWT(accountId: "account"), refreshToken: " "
            ))
        }
        let emptyClientId = OpenAICodexOAuthClient(
            configuration: .init(
                clientId: " ", authBaseURL: URL(string: "https://auth.example.com")!
            ),
            session: codexSession(),
            credentialStore: MemoryOAuthCredentialStore()
        )
        await #expect(throws: OpenAICodexOAuthError.self) {
            _ = try await emptyClientId.startDeviceAuthorization()
        }
        #expect(await router.recordedRequests().isEmpty)
    }

    private func makeCodexSessionRequest(
        promptCacheKey: String
    ) throws -> (body: JSONValue, request: URLRequest) {
        let call = CallOptions(
            model: "gpt-5-codex",
            prompt: [.user("hello")],
            providerOptions: ["openai-codex": ["prompt_cache_key": .string(promptCacheKey)]]
        )
        let client = AIClient(
            provider: .init(id: "codex", api: nil, speaking: .openAICodex),
            configuration: .init(authorization: .oauth(OAuthCredential(
                accessToken: codexJWT(accountId: "account")
            )))
        )
        let body = OpenAICodexResponsesRequest.encode(call).body
        return (body, try client.makeRequest(wire: .openAICodex, options: call, body: body))
    }
}

private func codexClient(
    session: URLSession,
    credentialProvider: any OAuthCredentialProviding
) -> AIClient {
    AIClient(
        provider: .init(id: "codex", api: nil, speaking: .openAICodex),
        configuration: .init(oauthCredentialProvider: credentialProvider),
        session: session
    )
}

private let codexDoneSSE = Data(
    "data: {\"type\":\"response.done\",\"sequence_number\":0,\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[]}}\n\ndata: [DONE]\n\n".utf8
)

private let codexTextDoneSSE = Data(
    (
        "data: {\"type\":\"response.output_text.delta\",\"sequence_number\":0,"
            + "\"item_id\":\"msg_1\",\"output_index\":0,\"content_index\":0,"
            + "\"delta\":\"hello\"}\n\n"
            + "data: {\"type\":\"response.done\",\"sequence_number\":1,"
            + "\"response\":{\"id\":\"resp_1\",\"status\":\"completed\",\"output\":[]}}\n\n"
            + "data: [DONE]\n\n"
    ).utf8
)

private func jsonData(_ value: JSONValue) -> Data {
    Data((try! value.encodedString()).utf8)
}

func codexJWT(accountId: String, expiresAt: Date? = nil) -> String {
    func encode(_ value: JSONValue) -> String {
        let data = Data((try! value.encodedString()).utf8)
        return data.base64EncodedString()
            .replacingOccurrences(of: "+", with: "-")
            .replacingOccurrences(of: "/", with: "_")
            .replacingOccurrences(of: "=", with: "")
    }
    let header: JSONValue = ["alg": "none"]
    var payload: [String: JSONValue] = [
        "https://api.openai.com/auth": ["chatgpt_account_id": .string(accountId)]
    ]
    if let expiresAt { payload["exp"] = .number(expiresAt.timeIntervalSince1970) }
    return "\(encode(header)).\(encode(.object(payload))).signature"
}

actor MemoryOAuthCredentialStore: OAuthCredentialStore {
    private var credential: OAuthCredential?
    private(set) var loadCount = 0
    private(set) var savedCredentials: [OAuthCredential?] = []

    init(_ credential: OAuthCredential? = nil) {
        self.credential = credential
    }

    func loadCredential(for key: String) async throws -> OAuthCredential? {
        loadCount += 1
        return credential
    }

    func saveCredential(_ credential: OAuthCredential?, for key: String) async throws {
        self.credential = credential
        savedCredentials.append(credential)
    }
}

actor RefreshProbe {
    private(set) var count = 0
    private let result: OAuthCredential

    init(result: OAuthCredential) {
        self.result = result
    }

    func refresh(_ credential: OAuthCredential) async throws -> OAuthCredential {
        count += 1
        return result
    }
}

actor SleepRecorder {
    private(set) var values: [TimeInterval] = []

    func record(_ value: TimeInterval) {
        values.append(value)
    }
}

final class DeviceCodeRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: OpenAICodexDeviceCode?

    var value: OpenAICodexDeviceCode? {
        lock.withLock { stored }
    }

    func record(_ code: OpenAICodexDeviceCode) {
        lock.withLock { stored = code }
    }
}

final class AuthorizationURLRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: URL?

    var value: URL? { lock.withLock { stored } }

    func record(_ url: URL) {
        lock.withLock { stored = url }
    }
}

final class LockedTestClock: @unchecked Sendable {
    private let lock = NSLock()
    private var date: Date
    private var recordedSleeps: [TimeInterval] = []

    init(_ date: Date) {
        self.date = date
    }

    func now() -> Date { lock.withLock { date } }

    func advance(by seconds: TimeInterval) {
        lock.withLock {
            recordedSleeps.append(seconds)
            date = date.addingTimeInterval(seconds)
        }
    }

    var sleeps: [TimeInterval] { lock.withLock { recordedSleeps } }
}

actor StartGate {
    private var continuation: CheckedContinuation<Void, Never>?
    private var released = false

    func wait() async {
        if released { return }
        await withCheckedContinuation { continuation in
            self.continuation = continuation
        }
    }

    func release() {
        released = true
        continuation?.resume()
        continuation = nil
    }
}

actor RefreshGate {
    private(set) var count = 0
    private let result: OAuthCredential
    private var entered: CheckedContinuation<Void, Never>?
    private var releaseContinuation: CheckedContinuation<Void, Never>?

    init(result: OAuthCredential) {
        self.result = result
    }

    func refresh(_ credential: OAuthCredential) async -> OAuthCredential {
        count += 1
        entered?.resume()
        entered = nil
        await withCheckedContinuation { continuation in
            releaseContinuation = continuation
        }
        return result
    }

    func waitUntilEntered() async {
        if count > 0 { return }
        await withCheckedContinuation { continuation in
            entered = continuation
        }
    }

    func release() {
        releaseContinuation?.resume()
        releaseContinuation = nil
    }
}

private extension Array {
    func asyncMap<T>(_ transform: (Element) async throws -> T) async rethrows -> [T] {
        var values: [T] = []
        values.reserveCapacity(count)
        for element in self { values.append(try await transform(element)) }
        return values
    }
}

struct CodexURLStub: Sendable {
    var status: Int
    var body: Data
    var headers: [String: String] = ["content-type": "text/event-stream"]
    var waitsForCancellation = false
    var waitsAfterBodyForCancellation = false
}

enum CodexURLProtocolError: Error {
    case missingStub
}

actor CodexURLRouter {
    private var stubs: [CodexURLStub] = []
    private var requests: [URLRequest] = []
    private var waiters: [(Int, CheckedContinuation<Void, Never>)] = []
    private var cancellations = 0
    private var cancellationWaiters: [(Int, CheckedContinuation<Void, Never>)] = []

    var cancellationCount: Int { cancellations }

    func configure(_ stubs: [CodexURLStub]) {
        self.stubs = stubs
        requests = []
        waiters = []
        cancellations = 0
        cancellationWaiters = []
    }

    func next(for request: URLRequest) throws -> CodexURLStub {
        requests.append(request)
        let count = requests.count
        let ready = waiters.filter { $0.0 <= count }
        waiters.removeAll { $0.0 <= count }
        for (_, continuation) in ready { continuation.resume() }
        guard !stubs.isEmpty else { throw CodexURLProtocolError.missingStub }
        return stubs.removeFirst()
    }

    func recordedRequests() -> [URLRequest] { requests }

    func waitForRequests(_ count: Int) async {
        if requests.count >= count { return }
        await withCheckedContinuation { continuation in
            waiters.append((count, continuation))
        }
    }

    func recordCancellation() {
        cancellations += 1
        let ready = cancellationWaiters.filter { $0.0 <= cancellations }
        cancellationWaiters.removeAll { $0.0 <= cancellations }
        for (_, continuation) in ready { continuation.resume() }
    }

    func waitForCancellations(_ count: Int) async {
        if cancellations >= count { return }
        await withCheckedContinuation { continuation in
            cancellationWaiters.append((count, continuation))
        }
    }
}

final class CodexURLProtocol: URLProtocol, @unchecked Sendable {
    static let router = CodexURLRouter()
    private var responseTask: Task<Void, Never>?

    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }

    override func startLoading() {
        let context = CodexURLProtocolContext(
            owner: self,
            request: request,
            client: client
        )
        responseTask = Task {
            await context.load()
        }
    }

    override func stopLoading() {
        responseTask?.cancel()
        let router = Self.router
        Task { await router.recordCancellation() }
    }
}

private final class CodexURLProtocolContext: @unchecked Sendable {
    private let owner: CodexURLProtocol
    private let request: URLRequest
    private weak var client: URLProtocolClient?

    init(owner: CodexURLProtocol, request: URLRequest, client: URLProtocolClient?) {
        self.owner = owner
        self.request = request
        self.client = client
    }

    func load() async {
        do {
            let capturedRequest = request.capturingHTTPBodyStream()
            let stub = try await CodexURLProtocol.router.next(for: capturedRequest)
            if stub.waitsForCancellation {
                try await Task.sleep(for: .seconds(60))
            }
            try Task.checkCancellation()
            guard let url = request.url,
                  let response = HTTPURLResponse(
                      url: url,
                      statusCode: stub.status,
                      httpVersion: "HTTP/1.1",
                      headerFields: stub.headers
                  )
            else { throw CodexURLProtocolError.missingStub }
            client?.urlProtocol(owner, didReceive: response, cacheStoragePolicy: .notAllowed)
            client?.urlProtocol(owner, didLoad: stub.body)
            if stub.waitsAfterBodyForCancellation {
                try await Task.sleep(for: .seconds(60))
            }
            try Task.checkCancellation()
            client?.urlProtocolDidFinishLoading(owner)
        } catch {
            client?.urlProtocol(owner, didFailWithError: error)
        }
    }
}

private extension URLRequest {
    func capturingHTTPBodyStream() -> URLRequest {
        guard httpBody == nil, let stream = httpBodyStream else { return self }
        var captured = self
        var data = Data()
        var buffer = [UInt8](repeating: 0, count: 1_024)
        stream.open()
        defer { stream.close() }
        while true {
            let count = stream.read(&buffer, maxLength: buffer.count)
            if count <= 0 { break }
            data.append(buffer, count: count)
        }
        captured.httpBody = data
        return captured
    }
}

func codexSession() -> URLSession {
    let configuration = URLSessionConfiguration.ephemeral
    configuration.protocolClasses = [CodexURLProtocol.self]
    return URLSession(configuration: configuration)
}
