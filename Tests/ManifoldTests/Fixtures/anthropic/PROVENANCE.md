# Fixture provenance

Recorded Anthropic Messages API streaming responses, vendored from the AI SDK.

- Source: https://github.com/vercel/ai
- Path: `packages/anthropic/src/__fixtures__`
- Upstream commit: `6a5bdffac`
- License: MIT (see upstream LICENSE)
- Files: 27

Each `.chunks.txt` contains one decoded SSE `data:` payload per line, in the
order the provider sent them.

Refresh with `Scripts/sync-fixtures.sh`. Re-run it after upgrading the pinned
upstream commit: a diff here is the earliest signal that a provider changed its
wire format.
