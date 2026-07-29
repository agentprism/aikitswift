# Fixture provenance

Recorded provider streaming responses, vendored from the AI SDK.

- Source: https://github.com/vercel/ai
- Upstream commit: `6a5bdffac`
- License: MIT (see upstream LICENSE)

Each `.chunks.txt` contains one decoded SSE `data:` payload per line,
in the order the provider sent them. A file may hold several sequential
API calls; each `message_start` (or equivalent) begins a new stream.

Sets are grouped by wire protocol rather than by vendor — a DeepSeek
recording exercises the same OpenAI Completions mapper a Groq recording
does.

| Set | Upstream path | Files |
|---|---|---|
| `anthropic` | `packages/anthropic/src` | 27 |
| `openai-completions` | `packages/openai/src/chat` | 2 |
| `openai-responses` | `packages/openai/src/responses` | 29 |
| `google` | `packages/google/src` | 20 |
| `deepseek` | `packages/deepseek/src` | 3 |
| `xai` | `packages/xai/src` | 7 |
| `groq` | `packages/groq/src` | 3 |
| `mistral` | `packages/mistral/src` | 3 |
| `cerebras` | `packages/cerebras/src` | 1 |
| `openai-compatible` | `packages/openai-compatible/src` | 2 |

**Total: 97 recorded streams.**

Refresh with `Scripts/sync-fixtures.sh`. Re-run it after bumping the
pinned upstream commit: a diff here is the earliest available signal
that a provider changed its wire format.
