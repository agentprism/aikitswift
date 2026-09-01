import AIKit
import Foundation

@main
struct ChatGPTHandoff {
    static func main() async throws {
        let anthropicKey = try requiredEnvironmentVariable("ANTHROPIC_API_KEY")
        let codexCredentials = try await logInToChatGPT()

        let ai = AIFacade(configurations: [
            "openai-codex": .init(oauthCredentialProvider: codexCredentials),
            "anthropic": .init(apiKey: anthropicKey),
        ])
        let codex = try ai.destination(
            providerId: "openai-codex",
            modelId: "gpt-5.4-mini"
        )
        let claude = try ai.destination(
            providerId: "anthropic",
            modelId: "claude-haiku-4-5"
        )

        var conversation: Prompt = [
            .system("Be concise. Preserve context when the conversation changes providers.")
        ]

        let first = try await ask(
            "What is 25 × 18? Explain the calculation briefly.",
            using: codex,
            in: &conversation,
            facade: ai
        )
        conversation.append(first.assistantMessage)

        let second = try await ask(
            "Is that calculation correct? If so, give one real-world example that uses it.",
            using: claude,
            in: &conversation,
            facade: ai
        )
        conversation.append(second.assistantMessage)
    }

    /// Uses a persisted Keychain credential when available. Otherwise it opens
    /// the ChatGPT browser login, receives the fixed loopback redirect, and
    /// installs the resulting credential into the refreshing request manager.
    private static func logInToChatGPT() async throws -> OAuthCredentialManager {
        let oauth = OpenAICodexOAuthClient()
        let credentials = oauth.credentialManager()

        do {
            _ = try await credentials.credential()
            print("Using the saved ChatGPT login.")
        } catch OAuthCredentialManagerError.missingCredential {
            print("Opening ChatGPT login in your browser…")
            let credential = try await oauth.authorizePKCE { url in
                openInDefaultBrowser(url)
            }
            try await credentials.install(credential)
            print("ChatGPT login saved in Keychain.")
        }

        return credentials
    }

    private static func ask(
        _ prompt: String,
        using destination: ModelDestination,
        in conversation: inout Prompt,
        facade: AIFacade
    ) async throws -> AIResponse {
        conversation.append(.user(prompt))
        print("\n[\(destination.providerId)/\(destination.modelId)]")

        var parts: [StreamPart] = []
        let request = AIRequest(
            destination: destination,
            prompt: conversation,
            thinking: .level(.medium)
        )

        for try await part in try facade.stream(request) {
            parts.append(part)
            if case .textDelta(_, let delta, _) = part {
                FileHandle.standardOutput.write(Data(delta.utf8))
            }
        }
        print()

        return AIResponse(parts: parts)
    }

    private static func requiredEnvironmentVariable(_ name: String) throws -> String {
        guard let value = ProcessInfo.processInfo.environment[name],
              !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else {
            throw ExampleError.missingEnvironmentVariable(name)
        }
        return value
    }

    private static func openInDefaultBrowser(_ url: URL) {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/open")
        process.arguments = [url.absoluteString]
        do {
            try process.run()
        } catch {
            print("Could not open a browser. Open this URL manually:\n\(url.absoluteString)")
        }
    }
}

private enum ExampleError: Error, LocalizedError {
    case missingEnvironmentVariable(String)

    var errorDescription: String? {
        switch self {
        case .missingEnvironmentVariable(let name):
            "Set \(name) before running this example."
        }
    }
}
