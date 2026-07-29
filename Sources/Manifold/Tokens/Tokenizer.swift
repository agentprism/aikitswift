import Foundation

/// Counts tokens in text.
///
/// Implementations fall into three tiers, and the difference matters:
///
/// 1. **Exact, remote** — the provider's own endpoint (Anthropic's
///    `count_tokens`, Google's `countTokens`). Authoritative, but costs a key
///    and a round trip.
/// 2. **Exact, local** — shipping the model family's real tokenizer. Precise,
///    but a large binary, and correct only for the family it came from.
/// 3. **Estimated, local** — ``HeuristicTokenizer``. Free and instant, wrong by
///    a margin, and the right default for a context breakdown.
///
/// - Warning: tokenizers are not interchangeable across model families. Using
///   OpenAI's tokenizer to count tokens for Claude under-reports by roughly
///   15–20% on prose and considerably more on code and non-English text.
///   Anthropic has also changed tokenizers between model generations, so even
///   "the Claude tokenizer" is version-specific.
public protocol Tokenizer: Sendable {
    /// Estimated or exact token count for a piece of text.
    func count(_ text: String) -> Int
}

/// A local, offline token estimator.
///
/// Counts by script rather than by byte, because the ratio differs sharply:
/// Latin text runs about four characters per token, while CJK runs closer to
/// one. A uniform `characters / 4` rule under-counts Chinese text by roughly
/// four times — which, for a context readout, is the difference between "half
/// full" and "over the limit".
///
/// Accurate to roughly ±15% on mixed content. Good enough to render a
/// breakdown; not good enough to decide whether a request will fit. For that,
/// calibrate against a real total (see ``ContextUsage/calibrated(toTotal:)``)
/// or ask the provider.
public struct HeuristicTokenizer: Tokenizer {

    /// Characters per token for alphabetic scripts.
    public var charactersPerToken: Double
    /// Tokens per character for CJK and other dense scripts.
    public var cjkTokensPerCharacter: Double

    public init(charactersPerToken: Double = 3.8, cjkTokensPerCharacter: Double = 1.0) {
        self.charactersPerToken = charactersPerToken
        self.cjkTokensPerCharacter = cjkTokensPerCharacter
    }

    public func count(_ text: String) -> Int {
        guard !text.isEmpty else { return 0 }

        var denseCharacters = 0
        var otherCharacters = 0

        for scalar in text.unicodeScalars {
            if Self.isDense(scalar) {
                denseCharacters += 1
            } else {
                otherCharacters += 1
            }
        }

        let dense = Double(denseCharacters) * cjkTokensPerCharacter
        let other = Double(otherCharacters) / charactersPerToken

        // Any non-empty text costs at least one token.
        return max(1, Int((dense + other).rounded()))
    }

    /// Scripts where a character is worth about a whole token.
    private static func isDense(_ scalar: Unicode.Scalar) -> Bool {
        switch scalar.value {
        case 0x3040...0x30FF,    // Hiragana, Katakana
             0x3400...0x4DBF,    // CJK Extension A
             0x4E00...0x9FFF,    // CJK Unified Ideographs
             0xAC00...0xD7AF,    // Hangul syllables
             0xF900...0xFAFF,    // CJK Compatibility Ideographs
             0x20000...0x2FA1F:  // CJK Extensions B–F
            true
        default:
            false
        }
    }
}
