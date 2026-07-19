namespace PredictiveCoding.LLM.Eval;

// ─────────────────────────────────────────────────────────────────────────────
// EvalCorpus  —  the tiny fixed yardstick text.
//
// Runtime datasets stay out of git (house rule); this deliberately small
// (~1 KB) corpus IS committed, so every perplexity number quoted in tests,
// demos and roadmap phases is measured against the same text, forever.
// ─────────────────────────────────────────────────────────────────────────────

public static class EvalCorpus
{
    /// <summary>The yardstick text, line by line (feed straight to StreamingTextDataset).</summary>
    public static readonly string[] TinyLines =
    [
        "The brain is a prediction machine.",
        "Every layer guesses what the layer below will report next.",
        "When the guess is wrong, the error climbs upward and the model adjusts.",
        "When the guess is right, silence: nothing to explain, nothing to send.",
        "Learning is the slow art of being surprised less often.",
        "A language model plays the same game with words.",
        "Given the words so far, it bets on the next one.",
        "Good bets make short codes, and short codes mean low perplexity.",
        "The brain is a prediction machine.",
        "Every layer guesses what the layer below will report next.",
        "Good bets make short codes, and short codes mean low perplexity.",
        "Learning is the slow art of being surprised less often.",
    ];

    /// <summary>The same corpus as a single newline-joined string.</summary>
    public static string Tiny => string.Join('\n', TinyLines);
}
