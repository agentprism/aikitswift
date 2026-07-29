#!/usr/bin/env bash
#
# Syncs streaming fixtures from the AI SDK repository.
#
# These are recorded provider responses: each .chunks.txt holds one decoded SSE
# `data:` payload per line, captured from a real API call. They are what lets
# this library be tested against real wire behaviour without holding an API key
# for a single provider.
#
# Upstream: https://github.com/vercel/ai (MIT)
#
# Usage:
#   Scripts/sync-fixtures.sh [path-to-vercel-ai-checkout]

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AI_SDK="${1:-$REPO_ROOT/../vercel-ai}"
FIXTURES="$REPO_ROOT/Tests/ManifoldTests/Fixtures"

if [ ! -d "$AI_SDK/packages" ]; then
    echo "error: no AI SDK checkout at $AI_SDK" >&2
    echo "       clone it first:" >&2
    echo "       git clone --filter=blob:none https://github.com/vercel/ai.git" >&2
    exit 1
fi

# Fixture set -> upstream package path.
#
# Grouped by the *wire protocol* under test, not by vendor: a DeepSeek recording
# exercises the same OpenAI Completions mapper that a Groq recording does, so
# they belong to the same set.
SETS=(
    "anthropic:packages/anthropic/src"
    "openai-completions:packages/openai/src/chat"
    "openai-responses:packages/openai/src/responses"
    "google:packages/google/src"
    "deepseek:packages/deepseek/src"
    "xai:packages/xai/src"
    "groq:packages/groq/src"
    "mistral:packages/mistral/src"
    "cerebras:packages/cerebras/src"
    "openai-compatible:packages/openai-compatible/src"
)

UPSTREAM_SHA="$(git -C "$AI_SDK" rev-parse --short HEAD 2>/dev/null || echo unknown)"
total=0

{
    echo "# Fixture provenance"
    echo
    echo "Recorded provider streaming responses, vendored from the AI SDK."
    echo
    echo "- Source: https://github.com/vercel/ai"
    echo "- Upstream commit: \`$UPSTREAM_SHA\`"
    echo "- License: MIT (see upstream LICENSE)"
    echo
    echo "Each \`.chunks.txt\` contains one decoded SSE \`data:\` payload per line,"
    echo "in the order the provider sent them. A file may hold several sequential"
    echo "API calls; each \`message_start\` (or equivalent) begins a new stream."
    echo
    echo "Sets are grouped by wire protocol rather than by vendor — a DeepSeek"
    echo "recording exercises the same OpenAI Completions mapper a Groq recording"
    echo "does."
    echo
    echo "| Set | Upstream path | Files |"
    echo "|---|---|---|"
} > "$FIXTURES/PROVENANCE.md"

for entry in "${SETS[@]}"; do
    name="${entry%%:*}"
    path="${entry#*:}"
    src="$AI_SDK/$path"

    [ -d "$src" ] || continue

    dest="$FIXTURES/$name"
    mkdir -p "$dest"
    rm -f "$dest"/*.chunks.txt

    count=0
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        # Flatten nested paths into a unique filename.
        cp "$f" "$dest/$(basename "$f")"
        count=$((count + 1))
    done < <(find "$src" -name "*.chunks.txt" 2>/dev/null)

    if [ "$count" -eq 0 ]; then
        rmdir "$dest" 2>/dev/null || true
        continue
    fi

    echo "| \`$name\` | \`$path\` | $count |" >> "$FIXTURES/PROVENANCE.md"
    echo "  $name: $count"
    total=$((total + count))
done

{
    echo
    echo "**Total: $total recorded streams.**"
    echo
    echo "Refresh with \`Scripts/sync-fixtures.sh\`. Re-run it after bumping the"
    echo "pinned upstream commit: a diff here is the earliest available signal"
    echo "that a provider changed its wire format."
} >> "$FIXTURES/PROVENANCE.md"

# Superseded by the per-set layout above.
rm -f "$FIXTURES/anthropic/PROVENANCE.md"

echo "synced $total fixtures from $UPSTREAM_SHA"
