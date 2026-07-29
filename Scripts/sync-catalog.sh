#!/usr/bin/env bash
#
# Syncs the provider catalog from an upstream checkout.
#
# The catalog is data, not code: each provider is a small JSON document naming
# its base URL, its models, and — crucially — the `adapter` that identifies
# which wire protocol it speaks. Fifty providers resolve to a handful of
# adapters, which is the whole reason this library is structured the way it is.
#
# Keeping the catalog as vendored JSON rather than hand-written Swift means
# model metadata (context window, tool support, whether temperature is even
# accepted) tracks upstream instead of rotting here.
#
# Usage:
#   Scripts/sync-catalog.sh <path-to-providers-directory>
#
# The source is a directory of per-provider JSON documents. Set
# AIKIT_CATALOG_SRC to avoid passing it every time.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-${AIKIT_CATALOG_SRC:-}}"
DEST="$REPO_ROOT/Sources/AIKit/Catalog/providers"

if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
    echo "error: pass a directory of provider JSON documents" >&2
    echo "       Scripts/sync-catalog.sh <path-to-providers-directory>" >&2
    echo "       (or set AIKIT_CATALOG_SRC)" >&2
    exit 1
fi

mkdir -p "$DEST"
rm -f "$DEST"/*.json

count=0
for f in "$SRC"/*.json; do
    [ -e "$f" ] || continue
    cp "$f" "$DEST/"
    count=$((count + 1))
done

UPSTREAM_SHA="$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)"

python3 - "$DEST" "$UPSTREAM_SHA" "$count" <<'PY'
import json, glob, os, sys, collections

dest, sha, count = sys.argv[1], sys.argv[2], sys.argv[3]
adapters = collections.Counter()
models = 0

for path in sorted(glob.glob(os.path.join(dest, "*.json"))):
    with open(path) as fh:
        data = json.load(fh)
    adapters[data.get("adapter", "?")] += 1
    models += len(data.get("models", []))

lines = [
    "# Provider catalog provenance",
    "",
    "Vendored provider definitions. Each file names a provider's base URL, its",
    "models, and the `adapter` identifying which wire protocol it speaks.",
    "",
    f"- Upstream commit: `{sha}`",
    f"- Providers: {count}",
    f"- Models: {models}",
    "",
    "## Adapters in use",
    "",
    "| Adapter | Providers |",
    "|---|---|",
]
for adapter, n in adapters.most_common():
    lines.append(f"| `{adapter}` | {n} |")
lines += [
    "",
    "This distribution is the argument for the whole architecture: many",
    "providers, few protocols. Implementing one adapter correctly serves every",
    "provider that points at it, and adding a provider is a config file that",
    "needs no new wire tests.",
    "",
    "Refresh with `Scripts/sync-catalog.sh`.",
]

with open(os.path.join(dest, os.pardir, "PROVENANCE.md"), "w") as fh:
    fh.write("\n".join(lines) + "\n")

print(f"  {count} providers, {models} models")
for adapter, n in adapters.most_common():
    print(f"    {adapter}: {n}")
PY

echo "synced catalog from $UPSTREAM_SHA -> $DEST"
