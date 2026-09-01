# ChatGPT → Anthropic handoff

A minimal command-line example of one conversation changing providers:

1. Load a saved ChatGPT credential or run OpenAI Codex browser login with PKCE.
2. Initialize one `AIFacade` with ChatGPT Codex and Anthropic configurations.
3. Prompt `gpt-5.6-sol` at `xhigh` and print normalized text deltas as they stream.
4. Append `response.assistantMessage`, switch to reasoning-enabled Claude Sonnet 4.5 at `high`, and continue streaming the same conversation.

ChatGPT access requires a compatible ChatGPT subscription. The OAuth credential is stored in the Apple Keychain; it is never written to this directory or printed. The second provider uses an Anthropic API key from the environment.

```sh
export ANTHROPIC_API_KEY="..."
cd Examples/ChatGPTHandoff
swift run ChatGPTHandoff
```

On the first run, the example opens the ChatGPT login page and listens for the fixed loopback callback on `localhost:1455`. Later runs load and refresh the saved credential automatically.

The handoff itself is only an append and a new destination:

```swift
conversation.append(first.assistantMessage)
conversation.append(.user("Check that result independently, then give the next solution."))

let continuation = AIRequest(
    destination: claude,
    prompt: conversation
)
```

Before the Claude request is encoded, AIKit converts foreign readable reasoning to text, removes Codex-only opaque state, remaps linked tool identifiers when necessary, and preserves the ordinary conversation content.
