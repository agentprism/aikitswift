import AIKit
import Darwin
import Foundation

@main
struct ChatGPTHandoff {
    static func main() async {
        do {
            try await run()
        } catch {
            let message = "ChatGPTHandoff failed: \(error.localizedDescription)\n"
            FileHandle.standardError.write(Data(message.utf8))
            exit(EXIT_FAILURE)
        }
    }

    private static func run() async throws {
        let anthropicKey = try requiredEnvironmentVariable("ANTHROPIC_API_KEY")
        let codexCredentials = try await logInToChatGPT()

        let ai = AIFacade(configurations: [
            "openai-codex": .init(oauthCredentialProvider: codexCredentials),
            "anthropic": .init(apiKey: anthropicKey),
        ])
        let codex = try ai.destination(
            providerId: "openai-codex",
            modelId: "gpt-5.6-sol"
        )
        let claude = try ai.destination(
            providerId: "anthropic",
            modelId: "claude-sonnet-4-5"
        )

        var conversation: Prompt = [
            .system("Be concise. Preserve context when the conversation changes providers.")
        ]

        let first = try await ask(
            "Find the smallest positive integer that is divisible by 7 and leaves remainder 1, 2, 3, and 4 when divided by 2, 3, 4, and 5 respectively. Explain briefly.",
            using: codex,
            thinking: .level(.xhigh),
            in: &conversation,
            facade: ai
        )
        conversation.append(first.assistantMessage)

        let second = try await ask(
            "Check that result independently, then give the next larger integer with the same properties.",
            using: claude,
            thinking: .level(.high),
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
            writeOutput("Using the saved ChatGPT login.\n")
        } catch OAuthCredentialManagerError.missingCredential {
            writeOutput("Opening ChatGPT login in your browser…\n")
            let credential = try await oauth.authorizePKCE { url in
                openInDefaultBrowser(url)
            }
            try await credentials.install(credential)
            writeOutput("ChatGPT login saved in Keychain.\n")
        }

        return credentials
    }

    private static func ask(
        _ prompt: String,
        using destination: ModelDestination,
        thinking: Thinking,
        in conversation: inout Prompt,
        facade: AIFacade
    ) async throws -> AIResponse {
        conversation.append(.user(prompt))
        let thinkingDescription = thinking.level?.rawValue ?? (thinking.isOff ? "off" : "on")
        writeOutput("\n[\(destination.providerId)/\(destination.modelId); thinking=\(thinkingDescription)]\n")

        var parts: [StreamPart] = []
        let request = AIRequest(
            destination: destination,
            prompt: conversation,
            thinking: thinking
        )

        for try await part in try facade.stream(request) {
            parts.append(part)
            if case .textDelta(_, let delta, _) = part {
                writeOutput(delta)
            }
        }
        writeOutput("\n")

        return AIResponse(parts: parts)
    }

    private static func writeOutput(_ text: String) {
        FileHandle.standardOutput.write(Data(text.utf8))
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
