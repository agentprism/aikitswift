# Manifold

Many LLM wire formats in, one normalized event stream out. Written in Swift, for Swift.

A manifold is the fitting that merges several inlets into a single outlet. That is
the whole job: Anthropic's Messages API, OpenAI's Completions and Responses APIs,
Google's Generative AI API and the rest all describe the same conversation in
mutually incompatible ways. Manifold maps them onto one event stream, so an app is
written once instead of once per vendor.

```swift
for try await part in stream {
    switch part {
    case .textDelta(_, let delta, _):      render(delta)
    case .reasoningDelta(_, let delta, _): renderThinking(delta)
    case .toolCall(let call):              try await execute(call)
    case .finish(let usage, _, _):         report(usage)
    default:                               break
    }
}
```

That loop does not change when the provider does.

## Status

Early. The Anthropic Messages protocol is implemented and conformance-tested; other
protocols are not written yet. The API will change.

| | |
|---|---|
| Normalized event spec | ✅ |
| SSE framing | ✅ |
| Anthropic Messages protocol | ✅ streaming |
| OpenAI Completions / Responses | ⬜ |
| Google Generative AI | ⬜ |
| Provider catalog | ⬜ |
| Request building | ⬜ (responses only, so far) |

## Install

```swift
.package(url: "https://github.com/zjywill/manifold.git", branch: "main")
```

Swift 6, macOS 14+, iOS 17+. No dependencies.

## Use

Manifold currently handles the response half: you build the request, it normalizes
what comes back.

```swift
import Manifold

var request = URLRequest(url: URL(string: "https://api.anthropic.com/v1/messages")!)
request.httpMethod = "POST"
request.setValue(apiKey, forHTTPHeaderField: "x-api-key")
request.setValue("2023-06-01", forHTTPHeaderField: "anthropic-version")
request.setValue("application/json", forHTTPHeaderField: "content-type")
request.httpBody = body   // your request JSON, with "stream": true

let (bytes, _) = try await URLSession.shared.bytes(for: request)

var wire = AnthropicMessagesWire()
for try await event in sseEvents(from: bytes) {
    for part in wire.map(rawJSON: event.data) {
        handle(part)
    }
}
```

One `AnthropicMessagesWire` per response — it is a state machine over a single
stream, not a long-lived client.

## The idea: protocols, not providers

The mistake to avoid is writing one implementation per vendor. There are roughly
forty providers worth supporting and only about eight wire formats between them,
because most providers speak someone else's protocol. Manifold splits along that
seam:

```
Sources/Manifold/
  Spec/        the normalized vocabulary — one enum every provider maps onto
  Wire/        one implementation per protocol   (~8, the real work)
  Providers/   one small config per provider     (~40, base URL + auth + models)
```

A provider is data: a base URL, an auth method, a model list, and a pointer to the
protocol it speaks. Adding one is a config file, not an implementation — and needs
no new wire tests, because the protocol it points at is already covered.

That factoring is borrowed from [pi-ai][pi], which arrived at it independently. The
counter-example is worth naming too: a well-known Swift LLM client keeps its
provider layer in a single 6,000-line file, and supports fewer formats for it.

## Testing without API keys

Every provider integration faces the same problem: you cannot test what you cannot
call, and nobody holds keys for forty vendors.

Manifold sidesteps it. The test suite replays **27 recorded provider responses**
captured from real API calls — vendored from the [AI SDK][aisdk] under MIT — and
asserts the normalized stream is well-formed. Recorded bytes in, expected events
out. No network, no credentials, no account.

```
$ swift test
Test run with 31 tests in 3 suites passed
```

The invariants checked on every recorded stream:

- text, reasoning and tool-input blocks form balanced `start → delta* → end` triads
- assembled tool-call arguments parse as JSON — fragments arrive individually
  invalid, so reassembly being off by one character is a real and silent failure
- `finish` appears exactly once, last, with internally consistent usage
- unrecognized chunks surface as `.raw` rather than vanishing, so a provider can
  ship a new event type without this library losing data

The fixtures cover the parts of the protocol that are easy to get wrong and hard to
discover: compaction, context editing, server-side tools, MCP, code execution,
fallbacks, and reasoning with signatures.

Refresh them with `Scripts/sync-fixtures.sh`. A diff there is the earliest available
signal that a provider changed its wire format.

Beyond fixtures, an Anthropic-compatible local server such as [Osaurus][osaurus]
gives real end-to-end coverage over a real socket, also without keys.

## Design notes

**Normalization follows the AI SDK.** The event vocabulary mirrors the AI SDK's
`LanguageModelV4StreamPart`, the most battle-tested normalization of this problem in
any ecosystem. Reusing its shape means its fixtures are directly usable as a
conformance suite, and its design review comes for free.

**Nothing is discarded.** Provider-specific detail that has no normalized home is
carried in `providerMetadata`, namespaced by provider. Unknown chunks pass through as
`.raw`. Raw usage payloads are preserved so a bill can be audited.

**Tool inputs stay strings.** `ToolCall.input` is the JSON *text* that streamed in,
not a parsed object, because re-encoding a parsed value would not reproduce the
original bytes. Parse at the point of use.

**`nil` means unreported.** Every usage field is optional and distinguishes "the
provider did not say" from zero.

**Key order is sorted on encode.** Prompt caching is a byte-level prefix match, so
unstable key ordering in a request body silently destroys every cache hit. This is
the kind of thing that costs money quietly rather than failing loudly.

## Prior art

- **[vercel/ai][aisdk]** — the normalized event spec, and the fixture corpus this
  library is tested against. MIT.
- **[pi-ai][pi]** — the protocol/provider split, and a reminder of how much of a
  provider layer is configuration rather than code.
- **[Osaurus][osaurus]** — Swift-native, and a usable local test target.

## License

MIT.

Recorded fixtures under `Tests/ManifoldTests/Fixtures/` are vendored from
[vercel/ai][aisdk] and remain under its MIT license. See
`Tests/ManifoldTests/Fixtures/anthropic/PROVENANCE.md`.

[aisdk]: https://github.com/vercel/ai
[pi]: https://github.com/earendil-works/pi
[osaurus]: https://github.com/osaurus-ai/osaurus
