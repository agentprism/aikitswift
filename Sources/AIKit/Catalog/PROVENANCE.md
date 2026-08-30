# Provider catalog provenance

Vendored provider definitions. Each file names a provider's base URL, its
models, and the `adapter` identifying which wire protocol it speaks.

- Upstream commit: `3e9d442`
- Providers: 185
- Models: 6200
- Updated from upstream: 185
- Added from upstream: 0
- Removed from previous catalog: 0
- Unsupported upstream providers omitted: 22

## Adapters in use

| Adapter | Providers |
|---|---|
| `openai` | 172 |
| `anthropic` | 10 |
| `gemini` | 2 |
| `openai-responses` | 1 |

This distribution is the argument for the whole architecture: many
providers, few protocols. Implementing one adapter correctly serves every
provider that points at it, and adding a provider is a config file that
needs no new wire tests.

Refresh with `Scripts/sync-catalog.sh`.
