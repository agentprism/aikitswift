# Provider catalog provenance

Vendored provider definitions. Each file names a provider's base URL, its
models, and the `adapter` identifying which wire protocol it speaks.

- Upstream commit: `9522fc231`
- Providers: 49
- Models: 413

## Adapters in use

| Adapter | Providers |
|---|---|
| `openai` | 38 |
| `anthropic` | 7 |
| `openai-responses` | 2 |
| `gemini` | 1 |
| `openai-codex` | 1 |

This distribution is the argument for the whole architecture: many
providers, few protocols. Implementing one adapter correctly serves every
provider that points at it, and adding a provider is a config file that
needs no new wire tests.

Refresh with `Scripts/sync-catalog.sh`.
