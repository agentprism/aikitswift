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
DEST="$REPO_ROOT/Tests/ManifoldTests/Fixtures/anthropic"

if [ ! -d "$AI_SDK/packages/anthropic/src/__fixtures__" ]; then
    echo "error: no AI SDK checkout at $AI_SDK" >&2
    echo "       clone it first:" >&2
    echo "       git clone --filter=blob:none https://github.com/vercel/ai.git" >&2
    exit 1
fi

mkdir -p "$DEST"
rm -f "$DEST"/*.chunks.txt

count=0
for f in "$AI_SDK"/packages/anthropic/src/__fixtures__/*.chunks.txt; do
    [ -e "$f" ] || continue
    cp "$f" "$DEST/"
    count=$((count + 1))
done

UPSTREAM_SHA="$(git -C "$AI_SDK" rev-parse --short HEAD 2>/dev/null || echo unknown)"

cat > "$DEST/PROVENANCE.md" <<EOF
# Fixture provenance

Recorded Anthropic Messages API streaming responses, vendored from the AI SDK.

- Source: https://github.com/vercel/ai
- Path: \`packages/anthropic/src/__fixtures__\`
- Upstream commit: \`$UPSTREAM_SHA\`
- License: MIT (see upstream LICENSE)
- Files: $count

Each \`.chunks.txt\` contains one decoded SSE \`data:\` payload per line, in the
order the provider sent them.

Refresh with \`Scripts/sync-fixtures.sh\`. Re-run it after upgrading the pinned
upstream commit: a diff here is the earliest signal that a provider changed its
wire format.
EOF

echo "synced $count fixtures from $UPSTREAM_SHA -> $DEST"
