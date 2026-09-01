import Foundation
import Testing

@testable import AIKit

@Suite("Provider catalog")
struct CatalogTests {

    @Test("the bundled catalog loads")
    func loads() {
        // Also guards the resource bundling: a packaging mistake would make
        // every other test here vacuously pass on an empty catalog.
        #expect(ProviderCatalog.all.count >= 40)
        #expect(ProviderCatalog.all.allSatisfy { !$0.id.isEmpty })
    }

    @Test("many providers resolve to few protocols")
    func manyProvidersFewProtocols() {
        let recognized = ProviderCatalog.all.filter { $0.wireProtocol != nil }
        let protocols = Set(recognized.compactMap(\.wireProtocol))

        // The ratio is the architecture in one assertion: implementing a
        // protocol serves every provider pointing at it.
        #expect(recognized.count >= 40)
        #expect(protocols.count <= WireProtocol.allCases.count)
        #expect(recognized.count > protocols.count * 5)
    }

    @Test("every adapter in the catalog is implemented")
    func allAdaptersImplemented() {
        // A new adapter upstream should surface here as a failing test rather
        // than as a provider that silently cannot be used.
        let unknown = Set(
            ProviderCatalog.all
                .compactMap(\.adapter)
                .filter { WireProtocol(rawValue: $0) == nil }
        )
        #expect(unknown.isEmpty, "unimplemented adapters: \(unknown.sorted())")
    }

    @Test("Chat Completions is the dominant protocol")
    func completionsDominates() {
        let completions = ProviderCatalog.providers(speaking: .openAICompletions)
        let others = ProviderCatalog.all.filter {
            $0.wireProtocol != nil && $0.wireProtocol != .openAICompletions
        }

        #expect(completions.count > others.count)
    }

    @Test("known providers resolve with their protocol")
    func resolvesKnownProviders() {
        #expect(ProviderCatalog.provider("anthropic")?.wireProtocol == .anthropicMessages)
        #expect(ProviderCatalog.provider("deepseek")?.wireProtocol == .openAICompletions)
        #expect(ProviderCatalog.provider("google")?.wireProtocol == .googleGenerativeAI)
        #expect(ProviderCatalog.provider("openai-codex")?.wireProtocol == .openAICodex)
        #expect(
            ProviderCatalog.provider("openai-codex")?.api
                == "https://chatgpt.com/backend-api"
        )
        #expect(
            ProviderCatalog.provider("google")?.api
                == "https://generativelanguage.googleapis.com"
        )
        #expect(ProviderCatalog.provider("nope-not-real") == nil)
    }

    @Test("model metadata carries the fields encoders depend on")
    func exposesModelMetadata() {
        let withLimits = ProviderCatalog.all
            .flatMap { $0.models ?? [] }
            .filter { $0.contextWindow != nil }

        // Context windows are the denominator of any context-usage report.
        #expect(withLimits.count >= 100)
        #expect(withLimits.allSatisfy { ($0.contextWindow ?? 0) > 0 })
    }

    @Test("models that reject temperature are flagged")
    func flagsTemperatureSupport() {
        // Newer Anthropic models reject `temperature` outright rather than
        // ignoring it. Encoders consult this to drop it and warn instead of
        // letting the request fail with a 400.
        let anthropic = ProviderCatalog.provider("anthropic")
        let rejecting = (anthropic?.models ?? []).filter { !$0.supportsTemperature }

        #expect(!rejecting.isEmpty, "expected at least one model that rejects temperature")
    }

    @Test("every protocol can produce a mapper")
    func makesMapperForEveryProtocol() {
        for wire in WireProtocol.allCases {
            var mapper = wire.makeMapper()
            // An empty chunk should be handled without crashing, and the
            // stream-start contract holds even for an unrecognized payload.
            let parts = mapper.map(chunk: .object([:]))
            #expect(parts.first != nil, "\(wire) produced nothing for an empty chunk")
        }
    }

    @Test("type-erased mapper preserves state across calls")
    func erasedMapperKeepsState() {
        // The enum wrapper writes state back on every call. If that were
        // dropped, every chunk would be mapped against a fresh state machine
        // and triads would never balance.
        var mapper = WireProtocol.openAICompletions.makeMapper()

        _ = mapper.map(chunk: ["id": "x", "choices": [["index": 0, "delta": ["content": "a"]]]])
        let second = mapper.map(chunk: ["choices": [["index": 0, "delta": ["content": "b"]]]])

        // A second textStart would mean the first chunk's state was lost.
        #expect(!second.contains { if case .textStart = $0 { return true } else { return false } })
        #expect(second.contains { if case .textDelta = $0 { return true } else { return false } })
    }
}
