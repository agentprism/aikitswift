import Foundation

/// How hard the model should think, in a vocabulary every provider can be
/// mapped onto.
///
/// Providers disagree about which rungs exist — some offer three, some six,
/// some only an on/off switch. A level is a *request*, not a promise: each
/// encoder clamps it to the nearest value the target model actually accepts.
public enum ThinkingLevel: String, Sendable, Hashable, Codable, CaseIterable, Comparable {
    case minimal
    case low
    case medium
    case high
    case xhigh
    case max

    var rank: Int {
        switch self {
        case .minimal: 0
        case .low: 1
        case .medium: 2
        case .high: 3
        case .xhigh: 4
        case .max: 5
        }
    }

    /// A token budget for providers that express depth as a number rather than
    /// a name — Anthropic's `budget_tokens`, Gemini's `thinkingBudget`.
    ///
    /// Clamped to the model's advertised range by the encoder; these are only
    /// the starting points for a model whose range is unknown.
    var budgetTokens: Int {
        switch self {
        case .minimal: 1024
        case .low: 4096
        case .medium: 8192
        case .high: 16384
        case .xhigh: 24576
        case .max: 32768
        }
    }

    public static func < (lhs: Self, rhs: Self) -> Bool { lhs.rank < rhs.rank }
}

/// What to ask a model to do about thinking.
///
/// The important case is ``off``. Several providers default reasoning *on* —
/// DeepSeek, Qwen, Gemini Flash, GLM — so "say nothing" is not the same as
/// "don't think": for a task that is not a reasoning problem, thinking tokens
/// are pure latency and cost. Turning it off takes three incompatible shapes on
/// the wire (`thinking: {type: disabled}`, `enable_thinking: false`,
/// `thinking_budget: 0`), which is exactly why it belongs here as data rather
/// than at each call site.
public enum Thinking: Sendable, Hashable {
    /// Explicitly disable thinking, in whichever shape the provider expects.
    case off
    /// Enable thinking at the model's own default depth.
    case on
    /// Enable thinking at a normalized depth.
    case level(ThinkingLevel)
    /// Enable thinking with an explicit token budget. Zero or less means ``off``.
    case budget(tokens: Int)

    /// Parses a settings-menu string: `off`/`none`/`disabled`, `on`/`auto`, or
    /// a level name. Returns `nil` for anything else.
    public init?(setting: String) {
        switch setting.trimmingCharacters(in: .whitespaces).lowercased() {
        case "off", "none", "disabled", "false", "0":
            self = .off
        case "on", "auto", "enabled", "adaptive", "true", "default":
            self = .on
        case let value:
            guard let level = ThinkingLevel(rawValue: value) else { return nil }
            self = .level(level)
        }
    }

    public var isOff: Bool {
        switch self {
        case .off: true
        case .budget(let tokens): tokens <= 0
        case .on, .level: false
        }
    }

    /// The requested level, where one was given.
    public var level: ThinkingLevel? {
        if case .level(let level) = self { return level }
        return nil
    }
}

// MARK: - Resolution

/// A ``Thinking`` request resolved against what one model can actually do.
///
/// Encoders consume this rather than re-deriving the same rules four times:
/// the shape of the request differs per protocol, but the decision — enable or
/// not, at what depth, and whether "off" is even expressible — does not.
struct ThinkingPlan: Sendable, Hashable {
    /// Whether to ask for thinking at all.
    var enabled: Bool
    /// Effort name in the *provider's* vocabulary, already clamped.
    var effort: String?
    /// Token budget, for providers that take one.
    var budgetTokens: Int?
    /// The effort value that means "off" for this model, where one exists —
    /// OpenAI's `none` (and `minimal` as the nearest approximation).
    var offEffort: String?
    /// Whether thinking can be turned off at all. `false` for models that
    /// always think; a warning has already been recorded in that case.
    var canDisable: Bool
    var warnings: [Warning]
}

extension Thinking {

    /// Resolves this request against a model's advertised capabilities.
    ///
    /// An unknown model is treated permissively: the request is rendered as
    /// asked rather than second-guessed, because the catalog not knowing a
    /// model says nothing about what that model supports.
    func plan(model: ModelInfo?, modelId: String) -> ThinkingPlan {
        let capability = model?.thinkingCapability
        var warnings: [Warning] = []

        func disable() -> ThinkingPlan {
            let canDisable = capability?.canDisable ?? true
            if !canDisable {
                warnings.append(Warning(
                    message: "\(modelId) always thinks; the request to turn thinking off was dropped.",
                    setting: "thinking"
                ))
            } else if capability?.disableIsApproximate == true {
                warnings.append(Warning(
                    message: "\(modelId) cannot stop thinking entirely; it was set to its lowest depth instead.",
                    setting: "thinking"
                ))
            }
            return ThinkingPlan(
                enabled: false,
                offEffort: capability?.offEffort,
                canDisable: canDisable,
                warnings: warnings
            )
        }

        switch self {
        case .off:
            return disable()

        case .budget(let tokens) where tokens <= 0:
            return disable()

        case .on:
            // A budget-only model has no "just think" switch — the number *is*
            // the switch — so plain `on` becomes a middling budget.
            let budget = capability.flatMap { capability in
                capability.effortValues.isEmpty ? capability.budget(for: .medium) : nil
            }
            return ThinkingPlan(
                enabled: true,
                budgetTokens: budget,
                canDisable: capability?.canDisable ?? true,
                warnings: warnings
            )

        case .level(let level):
            // An unknown model gets the level as asked; a known one gets the
            // nearest rung it actually accepts.
            guard let capability else {
                return ThinkingPlan(enabled: true, effort: level.rawValue, canDisable: true, warnings: warnings)
            }

            let effort = capability.nearestEffort(to: level)
            // Budget-only models (Anthropic 4.x, Gemini 2.5) need a number to
            // enable thinking at all, so the level is rendered as one.
            let budget = effort == nil ? capability.budget(for: level) : nil

            // A toggle-only model can be switched on but not tuned; say so
            // rather than inventing an effort name it would reject.
            if effort == nil, budget == nil, capability.supported {
                warnings.append(Warning(
                    message: "\(modelId) has no thinking-depth control; thinking was enabled at its default depth.",
                    setting: "thinking"
                ))
            }

            return ThinkingPlan(
                enabled: true,
                effort: effort,
                budgetTokens: budget,
                canDisable: capability.canDisable,
                warnings: warnings
            )

        case .budget(let tokens):
            let clamped = capability?.clampBudget(tokens) ?? tokens
            if clamped != tokens {
                warnings.append(Warning(
                    message: "\(modelId) accepts a thinking budget of \(capability?.budgetDescription ?? "") tokens; \(tokens) was clamped to \(clamped).",
                    setting: "thinking"
                ))
            }
            return ThinkingPlan(
                enabled: true,
                budgetTokens: clamped,
                canDisable: capability?.canDisable ?? true,
                warnings: warnings
            )
        }
    }
}
