# Manifold

Many LLM wire formats in, one normalized event stream out. Written in Swift, for Swift.

[简体中文](README.zh-CN.md)

A manifold is the fitting that merges several inlets into a single outlet. That is
the whole job: Anthropic's Messages API, OpenAI's Completions and Responses APIs and
Google's Generative AI API all describe the same conversation in mutually
incompatible ways. Manifold maps them onto one event stream, so an app is written
once instead of once per vendor.

```swift
let client = try ManifoldClient(providerId: "deepseek", configuration: .init(apiKey: key))

for try await part in try client.stream(CallOptions(model: "deepseek-v4-flash", prompt: [
    .system("You are terse."),
    .user("Weather in Paris?"),
])) {
    switch part {
    case .textDelta(_, let delta, _):      render(delta)
    case .reasoningDelta(_, let delta, _): renderThinking(delta)
    case .toolCall(let call):              try await execute(call)
    case .finish(let usage, _, _):         report(usage)
    default:                               break
    }
}
```

Change `"deepseek"` to `"anthropic"`, `"google"` or any of the other 46 providers.
Nothing else in that loop changes.

## The idea: protocols, not providers

The mistake to avoid is writing one implementation per vendor. The bundled catalog
holds **49 providers and 413 models — but only 5 wire protocols**, because most
providers speak someone else's:

| Protocol | Providers |
|---|---|
| OpenAI Chat Completions | 38 |
| Anthropic Messages | 7 |
| OpenAI Responses | 2 |
| OpenAI Codex | 1 |
| Google Generative AI | 1 |

Manifold splits along that seam:

```
Sources/Manifold/
  Spec/        the normalized vocabulary — one enum every provider maps onto
  Wire/        one implementation per protocol   (5, the real work)
  Providers/   the catalog                       (49 JSON configs, pure data)
  Tokens/      context attribution
  Client/      the plumbing between them
```

A provider is data: a base URL, an auth method, a model list, and the `adapter`
naming the protocol it speaks. Adding one is a config file, not an implementation —
and it needs no new wire tests, because the protocol it points at is already covered.

That factoring is borrowed from [pi-ai][pi] and the [dim-agent][dim] catalog, which
arrived at it independently. The counter-example is worth naming too: a well-known
Swift LLM client keeps its provider layer in a single 6,000-line file, and supports
fewer formats for it.

## Testing without API keys

Every provider integration faces the same problem: you cannot test what you cannot
call, and nobody holds keys for forty-nine vendors.

Manifold sidesteps it. The suite replays **97 recorded provider responses** captured
from real API calls — vendored from the [AI SDK][aisdk] under MIT — and asserts the
normalized stream is well-formed. Recorded bytes in, expected events out. No
network, no credentials, no account.

```
$ swift test
Test run with 100 tests in 10 suites passed
```

Fixtures are grouped by **protocol**, not vendor, so the Chat Completions mapper is
validated against real traffic from seven vendors at once:

| Set | Recordings |
|---|---|
| `anthropic` | 27 |
| `openai-responses` | 29 |
| `google` | 20 |
| `xai` | 7 |
| `deepseek`, `groq`, `mistral` | 3 each |
| `openai-completions`, `openai-compatible` | 2 each |
| `cerebras` | 1 |

Invariants checked on every recording:

- text, reasoning and tool-input blocks form balanced `start → delta* → end` triads
- assembled tool-call arguments parse as JSON — fragments arrive individually
  invalid, so reassembly being off by one character is a real and silent failure
- `finish` appears exactly once, last, with internally consistent usage
- unrecognized chunks surface as `.raw` rather than vanishing, so a provider can
  ship a new event type without this library losing data

Refresh with `Scripts/sync-fixtures.sh`. A diff there is the earliest available
signal that a provider changed its wire format.

Beyond fixtures, an Anthropic-compatible local server such as [Osaurus][osaurus]
gives real end-to-end coverage over a real socket, also without keys.

## Context attribution

Where a request's tokens went, and how much window is left:

```swift
let usage = ContextReporter().report(
    options,
    contextWindow: ProviderCatalog.model("claude-opus-4-8")?.1.contextWindow,
    extras: [("Skills", 5_500), ("Memory files", 284)]
)

for entry in usage.entries {
    print(entry.segment.label, ContextUsage.format(entry.tokens),
          String(format: "%.1f%%", usage.share(of: entry) * 100))
}
// Messages       463.4k  46.3%
// System prompt    6.1k   0.6%
// Skills           5.5k   0.6%
// Memory files      284   0.0%
```

Attribution is provider-independent; only counting is provider-specific, so the
tokenizer is injected. The default estimator is script-aware — Latin text runs about
four characters per token while CJK runs closer to one, and a uniform `characters/4`
rule under-counts Chinese by roughly fourfold.

For an exact total, don't pay for it twice:

```swift
usage.calibrated(toTotal: lastResponse.inputTokens.total ?? 0)
```

The previous response's usage is authoritative and already paid for. Anchoring to it
gives an exact total with proportionally-correct segments and no extra network call.
Reach for a provider's `count_tokens` endpoint only when the figure is needed
*before* sending.

## Install

```swift
.package(url: "https://github.com/zjywill/manifold.git", branch: "main")
```

Swift 6, macOS 14+, iOS 17+. No dependencies.

## Status

Early, and the API will change. Streaming responses and request encoding work across
all five protocols; the catalog covers 49 providers.

| | |
|---|---|
| Normalized event spec | ✅ |
| SSE framing | ✅ |
| Anthropic Messages | ✅ stream + request |
| OpenAI Chat Completions | ✅ stream + request |
| OpenAI Responses | ✅ stream + request |
| Google Generative AI | ✅ stream + request |
| Provider catalog | ✅ 49 providers, 413 models |
| Context attribution | ✅ |
| Non-streaming responses | ⬜ |
| Server-tool results (code exec, MCP, web search) | ⬜ surfaced as `.raw` |
| OAuth flows | ⬜ |

## Design notes

**Normalization follows the AI SDK.** The event vocabulary mirrors the AI SDK's
`LanguageModelV4StreamPart`, the most battle-tested normalization of this problem in
any ecosystem. Reusing its shape means its fixtures are directly usable as a
conformance suite, and its design review comes for free.

**Nothing is discarded.** Provider-specific detail that has no normalized home is
carried in `providerMetadata`, namespaced by provider. Unknown chunks pass through as
`.raw`. Raw usage payloads are preserved so a bill can be audited.

**Usage conventions disagree, and each one is encoded.** Three providers, three
different meanings for the same idea:

| | includes cached input? | includes reasoning in output? |
|---|---|---|
| Anthropic | no — add the cache legs back | n/a |
| OpenAI | yes — subtract to get uncached | yes — subtract to get text |
| Google | yes | **no** — add thoughts to get the total |

Applying one provider's arithmetic to another misprices silently rather than
failing. Each convention has its own test.

**Settings that would 400 are dropped and reported.** Newer Anthropic models reject
`temperature` outright. The catalog knows which, so the encoder drops it and emits a
`Warning` on `streamStart` instead of letting the request fail.

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
- **[dim-agent][dim]** — the provider catalog, including the Chinese providers the
  Western SDKs omit.
- **[Osaurus][osaurus]** — Swift-native, and a usable local test target.

## License

MIT.

Vendored data keeps its upstream license: recorded fixtures under
`Tests/ManifoldTests/Fixtures/` come from [vercel/ai][aisdk] (MIT), and the provider
catalog under `Sources/Manifold/Catalog/` from [dim-agent][dim]. See the
`PROVENANCE.md` in each directory.

[aisdk]: https://github.com/vercel/ai
[pi]: https://github.com/earendil-works/pi
[dim]: https://github.com/ThinkInAIXYZ
[osaurus]: https://github.com/osaurus-ai/osaurus
