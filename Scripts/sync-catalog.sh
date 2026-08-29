#!/usr/bin/env bash
#
# Syncs the provider catalog from a models.dev checkout or live API.
#
# Upstream moved from per-provider JSON files to TOML plus a generated
# api.json. This script pulls the latest catalog, converts it to the
# vendored JSON shape AIKit decodes, and preserves local-only providers.
#
# Usage:
#   Scripts/sync-catalog.sh <path-to-models.dev-checkout>
#
# Set AIKIT_CATALOG_SRC to avoid passing the path every time.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${1:-${AIKIT_CATALOG_SRC:-}}"
DEST="$REPO_ROOT/Sources/AIKit/Catalog/providers"

if [ -z "$SRC" ] || [ ! -d "$SRC" ]; then
    echo "error: pass a models.dev repository checkout" >&2
    echo "       Scripts/sync-catalog.sh <path-to-models.dev-checkout>" >&2
    echo "       (or set AIKIT_CATALOG_SRC)" >&2
    exit 1
fi

UPSTREAM_SHA="$(git -C "$SRC" rev-parse --short HEAD 2>/dev/null || echo unknown)"

API_JSON="$SRC/.sync/aikit-api.json"
mkdir -p "$SRC/.sync"

if [ -f "$SRC/packages/web/dist/_api.json" ]; then
    cp "$SRC/packages/web/dist/_api.json" "$API_JSON"
elif [ -f "$API_JSON" ] && [ "$(find "$API_JSON" -mmin -60 2>/dev/null)" ]; then
    :
else
    echo "fetching https://models.dev/api.json"
    curl -fsSL --max-time 120 "https://models.dev/api.json" -o "$API_JSON"
fi

python3 - "$SRC" "$API_JSON" "$DEST" "$UPSTREAM_SHA" <<'PY'
import json, os, sys, glob, collections, tomllib

src_root, api_path, dest, sha = sys.argv[1:5]

with open(api_path) as fh:
    upstream = json.load(fh)

ALIASES = {
    "moonshot": "moonshotai-cn",
    "moonshot-ai": "moonshotai",
    "qiniu": "qiniu-ai",
    "siliconflow-com": "siliconflow",
    "siliconflow": "siliconflow-cn",
    "novita": "novita-ai",
    "fireworks": "fireworks-ai",
    "lm-studio": "lmstudio",
    "vercel-ai-gateway": "vercel",
    "bailian-coding-plan": "alibaba-coding-plan-cn",
    "github": "github-copilot",
    "step-plan": "stepfun-step-plan",
}

PRESERVE_API = {
    "vercel-ai-gateway",
    "github",
    "custom-provider",
    "flymux-anthropic",
    "flymux-openai",
    "next-api-oauth",
    "openai-codex",
}

DEFAULT_APIS = {
    "anthropic": "https://api.anthropic.com",
    "openai": "https://api.openai.com/v1",
    "google": "https://generativelanguage.googleapis.com",
    "deepseek": "https://api.deepseek.com",
}


def read_provider_toml(provider_id):
    path = os.path.join(src_root, "providers", provider_id, "provider.toml")
    if not os.path.isfile(path):
        return {}
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def adapter_for(npm, provider_id, existing):
    if existing:
        return existing
    if provider_id == "openai-codex":
        return "openai-codex"
    if provider_id == "flymux-openai":
        return "openai-responses"
    if npm in ("@ai-sdk/anthropic", "@ai-sdk/google-vertex/anthropic"):
        return "anthropic"
    if npm in ("@ai-sdk/google", "@ai-sdk/google-vertex"):
        return "gemini"
    if npm == "@ai-sdk/openai" and provider_id == "openai":
        return "openai-responses"
    return "openai"


def infer_type(model):
    modalities = model.get("modalities") or {}
    inputs = set(modalities.get("input") or [])
    outputs = set(modalities.get("output") or [])
    label = f"{model.get('id', '')} {model.get('name', '')} {model.get('family', '')}".lower()

    if "realtime" in label:
        return "realtime"
    if outputs == {"image"}:
        return "image-generation"
    if "image" in outputs and any(token in label for token in ("image", "banana", "imagen", "dall")):
        return "image-generation"
    if outputs == {"audio"} and "text" in inputs:
        if any(token in label for token in ("tts", "speech", "voice")):
            return "speech-synthesis"
    if outputs == {"text"} and "audio" in inputs and any(
        token in label for token in ("asr", "transcrib", "whisper", "speech")
    ):
        return "speech-transcription"
    if "embedding" in label:
        return "other"
    return "chat"


def normalize_reasoning(value, previous):
    if isinstance(value, dict):
        return value
    if value is True:
        if isinstance(previous, dict):
            supported = previous.get("supported", True)
            default = previous.get("default", False)
            budget = previous.get("budget")
            out = {"supported": supported, "default": default}
            if budget is not None:
                out["budget"] = budget
            return out
        return {"supported": True, "default": False}
    if value is False:
        return {"supported": False}
    if isinstance(previous, dict):
        return previous
    return None


def transform_model(model, previous=None):
    previous = previous or {}
    out = {}
    for key, value in model.items():
        if key == "reasoning":
            normalized = normalize_reasoning(value, previous.get("reasoning"))
            if normalized is not None:
                out["reasoning"] = normalized
        else:
            out[key] = value

    name = out.get("name")
    if name:
        out.setdefault("display_name", name)
    out.setdefault("type", previous.get("type") or infer_type(out))
    return out


def provider_api(local_id, upstream_id, upstream_entry, existing, toml):
    if local_id in PRESERVE_API:
        return existing.get("api")
    if toml.get("api"):
        return toml["api"]
    if existing.get("api"):
        return existing["api"]
    if local_id in DEFAULT_APIS:
        return DEFAULT_APIS[local_id]
    return upstream_entry.get("api")


existing_paths = sorted(glob.glob(os.path.join(dest, "*.json")))
updated = 0
skipped = 0
adapters = collections.Counter()
model_count = 0

for path in existing_paths:
    with open(path) as fh:
        existing = json.load(fh)

    local_id = existing["id"]
    upstream_id = ALIASES.get(local_id, local_id)
    upstream_entry = upstream.get(upstream_id)

    if upstream_entry is None:
        skipped += 1
        adapters[existing.get("adapter", "?")] += 1
        model_count += len(existing.get("models", []))
        continue

    toml = read_provider_toml(upstream_id)
    npm = upstream_entry.get("npm") or toml.get("npm")
    previous_models = {m["id"]: m for m in existing.get("models", [])}

    models = []
    for model_id in sorted(upstream_entry.get("models", {})):
        models.append(
            transform_model(
                upstream_entry["models"][model_id],
                previous_models.get(model_id),
            )
        )

    provider = {
        "id": local_id,
        "name": upstream_entry.get("name") or existing.get("name"),
        "adapter": adapter_for(npm, local_id, existing.get("adapter")),
        "models": models,
    }

    api = provider_api(local_id, upstream_id, upstream_entry, existing, toml)
    if api:
        provider["api"] = api

    for key in ("doc", "display_name", "env", "npm", "reasoning_toggle"):
        if key in existing and key not in provider:
            provider[key] = existing[key]
        elif key in upstream_entry and key not in provider:
            provider[key] = upstream_entry[key]
        elif key in toml and key not in provider:
            provider[key] = toml[key]

    if local_id == "google" and not provider.get("api"):
        provider["api"] = DEFAULT_APIS["google"]

    with open(path, "w") as fh:
        json.dump(provider, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    updated += 1
    adapters[provider.get("adapter", "?")] += 1
    model_count += len(models)

count = len(existing_paths)
lines = [
    "# Provider catalog provenance",
    "",
    "Vendored provider definitions. Each file names a provider's base URL, its",
    "models, and the `adapter` identifying which wire protocol it speaks.",
    "",
    f"- Upstream commit: `{sha}`",
    f"- Providers: {count}",
    f"- Models: {model_count}",
    f"- Updated from upstream: {updated}",
    f"- Local-only (unchanged): {skipped}",
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

print(f"  {count} providers, {model_count} models ({updated} updated, {skipped} local-only)")
for adapter, n in adapters.most_common():
    print(f"    {adapter}: {n}")
PY

echo "synced catalog from $UPSTREAM_SHA -> $DEST"
