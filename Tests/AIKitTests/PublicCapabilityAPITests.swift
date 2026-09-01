import AIKit
import Testing

@Suite("Public model capability API")
struct PublicCapabilityAPITests {
    @Test("a normal import can declare and resolve custom model capabilities")
    func constructsCustomVisionModel() throws {
        let model = ModelInfo(
            id: "custom-vision-model",
            limit: ModelInfo.Limit(context: 128_000, output: 8_192),
            modalities: ModelInfo.Modalities(
                input: ["text", "image"],
                output: ["text"]
            )
        )
        let provider = ProviderInfo(
            id: "custom-provider",
            api: "https://custom.example",
            adapter: "openai",
            models: [model]
        )
        let facade = AIFacade(
            providers: [provider],
            configurations: ["custom-provider": .init(apiKey: "unused")]
        )

        let destination = try facade.destination(
            providerId: provider.id,
            modelId: model.id
        )
        let resolved = try facade.model(for: destination)

        #expect(resolved.modalities?.input == ["text", "image"])
        #expect(resolved.modalities?.output == ["text"])
        #expect(resolved.supportsVision)
        #expect(resolved.contextWindow == 128_000)
        #expect(resolved.maxOutputTokens == 8_192)
    }
}
