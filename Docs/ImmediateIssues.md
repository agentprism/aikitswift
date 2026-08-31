# Immediate implementation issues

This repository uses AIKit as its base package and retains AIKit's architecture:

- `Spec` defines the provider-neutral vocabulary.
- `Wire` implements protocol-specific request and response mapping.
- `Providers` contains provider and model configuration.
- `Client` connects the selected provider to its wire implementation.
- `Auth` contains provider authentication flows.

The provider-neutral facade and conversation semantics will use pi-ai as the reference implementation. The existing Swift proof of concept is not the target architecture and is not a requirements or test source. It is useful only as a prototype and as evidence that the observed ChatGPT Codex wire behavior can be implemented in Swift.

## Immediate priorities

### 1. Complete the provider-neutral facade

Provider switching must operate through AIKit's normalized architecture rather than through separate public provider clients.

The facade needs to:

- Select a model by provider and model identity.
- Route requests through the provider's declared wire protocol.
- Expose one normalized streaming and complete-response API.
- Keep provider wire types out of the provider-neutral public API.
- Record the provider, API, and model identity that produced assistant history.
- Preserve provider-specific state without making it part of the common vocabulary.

### 2. Add pi-ai-style conversation transformation

Before a prompt reaches a wire encoder, transform its history for the destination provider, API, model, and capabilities.

The transformer should follow pi-ai's behavior where Swift's strongly typed representation permits it:

- Retain opaque provider state only for an exact provider/API/model match.
- Convert reasoning to readable text when opaque state cannot be replayed safely.
- Remove provider metadata that does not belong to the destination.
- Normalize or remap tool-call identifiers for the destination protocol.
- Close orphaned tool calls with a synthetic result containing exactly `No result provided`.
- Remove aborted or failed assistant turns that must not be replayed.
- Downgrade unsupported user images to `(image omitted: model does not support images)`.
- Downgrade unsupported tool-result images to `(tool image omitted: model does not support images)`.

This belongs in the shared request path so every wire implementation receives already-valid destination history.

### 3. Correct OpenAI Responses history and event handling

The current Responses implementation handles ordinary text and tools but loses state needed for exact multi-turn behavior.

Immediate corrections include:

- Preserve and replay encrypted reasoning content.
- Preserve optional output item identifiers.
- Preserve assistant message item ID, status, and phase.
- Preserve function-call item identity separately from `call_id`.
- Support function namespaces.
- Support structured function outputs, including text and image content.
- Preserve known provider event payloads instead of retaining raw JSON only for unknown event types.
- Surface malformed known events as errors rather than silently returning no events.
- Preserve terminal response and provider error details.

Unknown valid events should continue to pass through without being discarded.

### 4. Finish OpenAI Codex as an AIKit provider

`WireProtocol.openAICodex` currently shares the ordinary OpenAI Responses endpoint, request encoding, event mapping, and generic bearer authentication. It must become a complete implementation within AIKit's existing provider/wire/client/auth structure.

The implementation needs:

- The ChatGPT Codex Responses endpoint.
- Codex-specific request fields and defaults.
- Required Codex headers and account identity.
- Device-code and PKCE authorization flows where applicable.
- Credential persistence and expiry handling.
- Coalesced credential refresh.
- One refresh and request replay after an authentication failure.
- Correct Codex SSE event mapping.
- Validation of known Codex events while preserving unknown valid events.

Shared Responses behavior should be factored into reusable helpers where the protocols genuinely match. Codex-specific behavior should remain explicit rather than being hidden behind ordinary Responses assumptions.

### 5. Correct Anthropic opaque-history replay

Anthropic streaming currently captures redacted-thinking data, but request encoding does not reconstruct a `redacted_thinking` block from that state.

The Anthropic path needs to:

- Replay signed thinking only when destination identity permits it.
- Replay redacted-thinking blocks in their original wire shape.
- Preserve required thinking signatures byte-for-byte.
- Group and order tool results according to Anthropic Messages requirements after shared history transformation.

## Validation for these changes

Changes should be covered by focused tests in this repository using AIKit's existing fixture-driven wire-testing approach, provider documentation, captured provider traffic where appropriate, and pi-ai behavior for facade and history semantics.

At minimum, validation should cover:

- Same-model opaque reasoning replay.
- Cross-model and cross-provider reasoning downgrade.
- OpenAI item identity and encrypted-reasoning round trips.
- Structured tool-result round trips.
- Malformed known events and unknown valid events.
- Codex authentication, refresh, retry, request, and SSE behavior.
- Anthropic signed and redacted-thinking replay.
- Provider switching through the common facade.

## Deliberately deferred

These are not immediate adoption blockers:

- Replacing AIKit's nested `AsyncThrowingStream` implementation.
- Introducing a package-wide transport protocol in place of `URLSession` injection.
- Changing the generic `extraHeaders` override policy.
- Auditing and replacing every unrelated `try?` fallback.

They should be revisited only in response to a demonstrated correctness problem, a concrete API requirement, or work already touching the affected path.

## Upstream synchronization

The repository maintains two long-lived branches:

- `upstream-sync` tracks `zjywill/aikitswift`'s `main` branch without project changes.
- `main` contains this repository's implementation.

Upstream changes are fetched and reviewed manually before being merged or selectively applied. Catalog, fixture, and dialect updates may be synchronized independently when that produces a smaller and more auditable change than merging upstream implementation work.
