import Foundation

/// What a model can do, and what it costs to talk to.
///
/// Sourced from the vendored catalog rather than hardcoded, so capability
/// changes track upstream instead of rotting in Swift.
public struct ModelInfo: Sendable, Hashable, Codable {
    public struct Reasoning: Sendable, Hashable, Codable {
        /// A token range for models whose thinking depth is a number.
        public struct Budget: Sendable, Hashable, Codable {
            public var min: Int?
            public var max: Int?
            /// The provider's own default. `-1` means "the model decides".
            public var `default`: Int?

            public init(min: Int? = nil, max: Int? = nil, default defaultValue: Int? = nil) {
                self.min = min
                self.max = max
                self.default = defaultValue
            }
        }

        public var supported: Bool?
        /// Whether reasoning is on when the request says nothing.
        ///
        /// The field that makes an explicit *off* switch necessary: where this
        /// is `true` — DeepSeek, Qwen, GLM, Gemini Flash — saying nothing buys
        /// thinking tokens whether the task needs them or not.
        public var `default`: Bool?
        public var budget: Budget?

        public init(supported: Bool? = nil, default defaultValue: Bool? = nil, budget: Budget? = nil) {
            self.supported = supported
            self.default = defaultValue
            self.budget = budget
        }
    }

    /// One way a model's thinking can be controlled.
    ///
    /// Kept as loose data because the vocabulary is upstream's, not ours: a
    /// `type` this library has not seen yet decodes and is ignored rather than
    /// failing the model.
    public struct ReasoningOption: Sendable, Hashable, Codable {
        /// `toggle`, `effort`, or `budget_tokens`.
        public var type: String?
        /// Effort names this model accepts, for `type == "effort"`.
        public var values: [String]?
        public var min: Int?
        public var max: Int?

        public init(type: String? = nil, values: [String]? = nil, min: Int? = nil, max: Int? = nil) {
            self.type = type
            self.values = values
            self.min = min
            self.max = max
        }
    }

    public struct Limit: Sendable, Hashable, Codable {
        /// Context window, in tokens. The denominator for a context-usage report.
        public var context: Int?
        /// Maximum output tokens per request.
        public var output: Int?
    }

    public struct Modalities: Sendable, Hashable, Codable {
        public var input: [String]?
        public var output: [String]?
    }

    public var id: String
    public var name: String?
    public var description: String?
    public var family: String?

    /// Whether files can be attached to a prompt.
    public var attachment: Bool?
    public var reasoning: Reasoning?
    /// How this model's thinking can be controlled — toggled, tuned by effort,
    /// or given a token budget. Often several at once.
    public var reasoningOptions: [ReasoningOption]?
    public var toolCall: Bool?
    public var structuredOutput: Bool?

    /// Whether the model accepts a sampling temperature at all.
    ///
    /// `false` on newer Anthropic models, which reject the parameter outright
    /// rather than ignoring it. Encoders consult this to drop the setting and
    /// warn instead of letting the request fail with a 400.
    public var temperature: Bool?

    public var limit: Limit?
    public var modalities: Modalities?
    public var openWeights: Bool?
    public var knowledge: String?
    public var releaseDate: String?
    public var lastUpdated: String?

    public init(
        id: String,
        name: String? = nil,
        description: String? = nil,
        family: String? = nil,
        attachment: Bool? = nil,
        reasoning: Reasoning? = nil,
        reasoningOptions: [ReasoningOption]? = nil,
        toolCall: Bool? = nil,
        structuredOutput: Bool? = nil,
        temperature: Bool? = nil,
        limit: Limit? = nil,
        modalities: Modalities? = nil,
        openWeights: Bool? = nil,
        knowledge: String? = nil,
        releaseDate: String? = nil,
        lastUpdated: String? = nil
    ) {
        self.id = id
        self.name = name
        self.description = description
        self.family = family
        self.attachment = attachment
        self.reasoning = reasoning
        self.reasoningOptions = reasoningOptions
        self.toolCall = toolCall
        self.structuredOutput = structuredOutput
        self.temperature = temperature
        self.limit = limit
        self.modalities = modalities
        self.openWeights = openWeights
        self.knowledge = knowledge
        self.releaseDate = releaseDate
        self.lastUpdated = lastUpdated
    }

    public var contextWindow: Int? { limit?.context }
    public var maxOutputTokens: Int? { limit?.output }
    public var supportsTools: Bool { toolCall ?? false }
    public var supportsReasoning: Bool { reasoning?.supported ?? false }
    /// Newer models reject `temperature`; absence of the flag means unknown, so
    /// it is treated as accepted.
    public var supportsTemperature: Bool { temperature ?? true }
    public var supportsVision: Bool { modalities?.input?.contains("image") ?? false }
}

// MARK: - Thinking

extension ModelInfo {

    /// What can actually be asked of this model's thinking.
    ///
    /// Flattened from the catalog's several overlapping fields so encoders ask
    /// one question — "can this be turned off, and how deep can it go?" —
    /// instead of re-deriving the answer from raw JSON four times.
    public struct ThinkingCapability: Sendable, Hashable {
        public var supported: Bool
        /// Whether thinking happens when the request says nothing.
        public var defaultOn: Bool
        /// Whether the model advertises an explicit on/off switch.
        public var hasToggle: Bool
        /// Effort names this model accepts, in the provider's own vocabulary.
        public var effortValues: [String]
        public var budgetRange: ClosedRange<Int>?

        /// Whether an explicit *off* can be honoured.
        ///
        /// A model that thinks by default and offers neither a toggle, a zero
        /// budget, nor a `none` effort cannot be quieted — Claude Fable 5 and
        /// `deepseek-reasoner` are the shape of this. Anything the catalog does
        /// not describe is assumed disablable, since defaulting to "you cannot"
        /// would silently keep paying for thinking.
        public var canDisable: Bool {
            guard supported, defaultOn else { return true }
            return hasToggle || offEffort != nil || (budgetRange?.lowerBound == 0)
        }

        /// The effort value that means "don't think", where the vocabulary has
        /// one. `minimal` is the nearest approximation where `none` is absent.
        public var offEffort: String? {
            if effortValues.contains("none") { return "none" }
            if effortValues.contains("minimal") { return "minimal" }
            return nil
        }

        /// Whether "off" can only be approximated.
        ///
        /// Gemini 3 Flash is the case: its floor is `minimal`, not silence, so
        /// the request is honoured as closely as the model allows and the
        /// caller is told the difference.
        public var disableIsApproximate: Bool {
            guard canDisable, supported, defaultOn else { return false }
            if hasToggle || budgetRange?.lowerBound == 0 { return false }
            return offEffort == "minimal"
        }

        /// The accepted effort name closest to a requested level.
        ///
        /// Ties go to the cheaper rung, and `none` is never chosen — asking to
        /// think harder must not turn thinking off.
        public func nearestEffort(to level: ThinkingLevel) -> String? {
            let candidates = effortValues.compactMap { value in
                ThinkingLevel(rawValue: value).map { ($0, value) }
            }
            guard !candidates.isEmpty else { return nil }

            return candidates.min { lhs, rhs in
                let left = abs(lhs.0.rank - level.rank)
                let right = abs(rhs.0.rank - level.rank)
                return left == right ? lhs.0.rank < rhs.0.rank : left < right
            }?.1
        }

        /// A token budget for a requested level, clamped to the model's range.
        public func budget(for level: ThinkingLevel) -> Int? {
            guard budgetRange != nil else { return nil }
            return clampBudget(level.budgetTokens)
        }

        public func clampBudget(_ tokens: Int) -> Int {
            guard let range = budgetRange else { return tokens }
            return Swift.min(Swift.max(tokens, range.lowerBound), range.upperBound)
        }

        var budgetDescription: String {
            guard let range = budgetRange else { return "any" }
            return "\(range.lowerBound)–\(range.upperBound)"
        }
    }

    public var thinkingCapability: ThinkingCapability {
        let options = reasoningOptions ?? []

        var efforts: [String] = []
        var lower: Int?
        var upper: Int?
        var hasToggle = false

        for option in options {
            switch option.type {
            case "effort":
                efforts.append(contentsOf: option.values ?? [])
            case "budget_tokens":
                lower = option.min ?? lower
                upper = option.max ?? upper
            case "toggle":
                hasToggle = true
            default:
                break
            }
        }

        // Some providers describe the budget on `reasoning` instead of as an
        // option; either is authoritative.
        lower = lower ?? reasoning?.budget?.min
        upper = upper ?? reasoning?.budget?.max

        var range: ClosedRange<Int>?
        if lower != nil || upper != nil {
            let low = lower ?? 0
            // No advertised ceiling: the model's output cap is the real bound,
            // since thinking is drawn from the same budget.
            let high = Swift.max(upper ?? maxOutputTokens ?? 32768, low)
            range = low...high
        }

        return ThinkingCapability(
            supported: reasoning?.supported ?? !options.isEmpty,
            defaultOn: reasoning?.default ?? false,
            hasToggle: hasToggle,
            effortValues: efforts,
            budgetRange: range
        )
    }
}

/// A provider: where to send a request, how to authenticate, and — the field
/// that matters most — which wire protocol it speaks.
public struct ProviderInfo: Sendable, Hashable, Codable {

    /// A provider-level override for how thinking is switched on and off.
    ///
    /// Resellers of the Anthropic protocol rename the values — MiniMax wants
    /// `adaptive` where Anthropic wants `enabled` — so the field path and both
    /// values travel as data rather than as a branch per provider.
    public struct ReasoningToggle: Sendable, Hashable, Codable {
        /// Dotted path into the request body, e.g. `thinking.type`.
        public var field: String
        public var enabled: String?
        public var disabled: String?

        public init(field: String, enabled: String? = nil, disabled: String? = nil) {
            self.field = field
            self.enabled = enabled
            self.disabled = disabled
        }

        /// Writes `value` at this toggle's path, creating intermediate objects.
        func apply(_ value: String, to body: inout [String: JSONValue]) {
            let path = field.split(separator: ".").map(String.init)
            guard !path.isEmpty else { return }
            body = Self.write(.string(value), at: path[...], into: body)
        }

        private static func write(
            _ value: JSONValue,
            at path: ArraySlice<String>,
            into object: [String: JSONValue]
        ) -> [String: JSONValue] {
            guard let key = path.first else { return object }

            var object = object
            if path.count == 1 {
                object[key] = value
            } else {
                let nested = object[key]?.objectValue ?? [:]
                object[key] = .object(write(value, at: path.dropFirst(), into: nested))
            }
            return object
        }
    }

    public var id: String
    public var name: String?
    /// Base URL. Absent for providers whose endpoint is configured per install
    /// (self-hosted gateways, local runtimes).
    public var api: String?
    /// The wire protocol, as named upstream.
    ///
    /// Kept as a raw string so an adapter this library has not implemented yet
    /// still loads instead of failing the whole catalog.
    public var adapter: String?
    /// How this provider spells "think" and "don't", when it differs from the
    /// protocol's own spelling.
    public var reasoningToggle: ReasoningToggle?
    public var models: [ModelInfo]?

    /// Memberwise, and public — an app that lets a user point at an arbitrary
    /// OpenAI-compatible endpoint has to be able to build one of these without
    /// finding it in the catalog first.
    public init(
        id: String,
        name: String? = nil,
        api: String? = nil,
        adapter: String? = nil,
        reasoningToggle: ReasoningToggle? = nil,
        models: [ModelInfo]? = nil
    ) {
        self.id = id
        self.name = name
        self.api = api
        self.adapter = adapter
        self.reasoningToggle = reasoningToggle
        self.models = models
    }

    /// Convenience for a user-configured endpoint: id, URL, and the protocol
    /// it speaks.
    public init(id: String, name: String? = nil, api: String?, speaking wire: WireProtocol) {
        self.init(id: id, name: name, api: api, adapter: wire.rawValue)
    }

    /// The implemented protocol for this provider, if there is one.
    public var wireProtocol: WireProtocol? {
        adapter.flatMap(WireProtocol.init(rawValue:))
    }

    public func model(_ id: String) -> ModelInfo? {
        models?.first { $0.id == id }
    }
}

/// The bundled provider catalog.
///
/// Loaded once, lazily, from JSON shipped with the package. Adding a provider
/// upstream needs no Swift changes here at all.
public enum ProviderCatalog {

    /// Every provider in the catalog, sorted by id.
    public static let all: [ProviderInfo] = load()

    private static let byId: [String: ProviderInfo] = Dictionary(
        all.map { ($0.id, $0) },
        uniquingKeysWith: { first, _ in first }
    )

    public static func provider(_ id: String) -> ProviderInfo? { byId[id] }

    /// Providers speaking a given protocol.
    public static func providers(speaking wire: WireProtocol) -> [ProviderInfo] {
        all.filter { $0.wireProtocol == wire }
    }

    /// Finds a model by id, searching every provider.
    ///
    /// Model ids are not globally unique — several providers resell the same
    /// model — so this returns the first match. Pass a provider id when the
    /// distinction matters.
    public static func model(_ modelId: String, provider providerId: String? = nil) -> (ProviderInfo, ModelInfo)? {
        let candidates = providerId.flatMap { byId[$0].map { [$0] } } ?? all

        for provider in candidates {
            if let model = provider.model(modelId) { return (provider, model) }
        }
        return nil
    }

    /// Whether the catalog actually loaded.
    ///
    /// `false` means the resource bundle was not found — almost always a
    /// packaging problem in the host app rather than anything about the
    /// catalog. An app that depends on the catalog should check this at launch
    /// and show ``diagnostics`` rather than presenting an empty provider list.
    public static var isLoaded: Bool { !all.isEmpty }

    /// The name SwiftPM gives the resource bundle: `<package>_<target>.bundle`.
    ///
    /// A packaging script has to copy this into `Contents/Resources`; it is
    /// spelled out here so the script and the loader cannot drift apart.
    public static let bundleName = "AIKitSwift_AIKit.bundle"

    /// Where the loader looked, for when it found nothing.
    public static var diagnostics: String {
        if let directory = catalogDirectory {
            return "Catalog loaded from \(directory.path) (\(all.count) providers)."
        }
        let searched = searchRoots.map(\.path).joined(separator: "\n  ")
        return """
            Could not find \(bundleName). Copy it into the app's Resources \
            directory alongside the executable. Searched:
              \(searched)
            """
    }

    private static func load() -> [ProviderInfo] {
        guard let directory = catalogDirectory,
              let files = try? FileManager.default.contentsOfDirectory(
                  at: directory, includingPropertiesForKeys: nil
              )
        else {
            return []
        }

        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase

        return files
            .filter { $0.pathExtension == "json" }
            .compactMap { url in
                guard let data = try? Data(contentsOf: url) else { return nil }
                // A provider whose schema drifted is skipped rather than
                // taking the whole catalog down with it.
                return try? decoder.decode(ProviderInfo.self, from: data)
            }
            .sorted { $0.id < $1.id }
    }

    // MARK: - Locating the resources

    /// Anchor for `Bundle(for:)`, which needs a class.
    private final class BundleAnchor {}

    /// Every plausible directory the resource bundle could sit in.
    ///
    /// SwiftPM's generated `Bundle.module` is deliberately not used: when the
    /// bundle is missing it calls `fatalError`, so a packaging mistake in the
    /// host app crashes on first catalog access instead of degrading. Searching
    /// by hand costs a few `FileManager` probes once and fails as `nil`.
    private static var searchRoots: [URL] {
        var roots: [URL] = []
        let anchor = Bundle(for: BundleAnchor.self)

        func add(_ url: URL?) {
            guard let url, !roots.contains(url) else { return }
            roots.append(url)
        }

        // A packaged app: Contents/Resources.
        add(Bundle.main.resourceURL)
        // A command-line tool: the directory holding the executable.
        add(Bundle.main.bundleURL)
        // Linked as a framework, or running from a test bundle.
        add(anchor.resourceURL)
        add(anchor.bundleURL)
        // The build directory, when the anchor is inside an .xctest bundle.
        add(anchor.bundleURL.deletingLastPathComponent())
        // A framework embedded in an app.
        add(Bundle.main.resourceURL?.deletingLastPathComponent()
            .appending(path: "Frameworks/AIKit.framework/Resources"))

        return roots
    }

    private static let catalogDirectory: URL? = {
        let manager = FileManager.default

        func providers(in directory: URL) -> URL? {
            let candidate = directory.appending(path: "Catalog/providers")
            var isDirectory: ObjCBool = false
            guard manager.fileExists(atPath: candidate.path, isDirectory: &isDirectory),
                  isDirectory.boolValue
            else { return nil }
            return candidate
        }

        for root in searchRoots {
            let bundleURL = root.appending(path: bundleName)
            // `Bundle(url:)` resolves the platform's layout — flat on Linux and
            // iOS, Contents/Resources on macOS.
            if let bundle = Bundle(url: bundleURL),
               let resources = bundle.resourceURL,
               let directory = providers(in: resources) {
                return directory
            }
            if let directory = providers(in: bundleURL) {
                return directory
            }
            // Resources flattened straight into the app, without the wrapper.
            if let directory = providers(in: root) {
                return directory
            }
        }

        return nil
    }()
}
