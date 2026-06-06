namespace PredictiveCoding.LLM.Summarization;

// ─────────────────────────────────────────────────────────────────────────────
// SummaryOptions  —  knobs for the extractive summariser.
//
// Mirrors RIASSUNTAZZO's SummaryOptions (MaxSentences + Ratio) and adds the extra
// controls that make this implementation self-contained and a notch better:
// a pluggable scoring method and an MMR-style redundancy filter.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>How sentences are scored before the top ones are kept.</summary>
public enum SummaryMethod
{
    /// <summary>Cosine similarity of each sentence's TF-IDF vector to the document centroid (default, robust).</summary>
    Centroid,

    /// <summary>PageRank over the sentence-similarity graph — unsupervised TextRank / LexRank.</summary>
    TextRank,

    /// <summary>Length-normalised raw term frequency — baseline, parity with RIASSUNTAZZO's scorer.</summary>
    TermFrequency,
}

/// <summary>Knobs for <see cref="ExtractiveSummarizer"/>.</summary>
public sealed record SummaryOptions
{
    /// <summary>Hard cap on the number of sentences kept (≤ 0 ⇒ governed only by <see cref="Ratio"/>).</summary>
    public int MaxSentences { get; init; } = 3;

    /// <summary>Fraction of the document's sentences to keep (combined with <see cref="MaxSentences"/>).</summary>
    public double Ratio { get; init; } = 0.3;

    /// <summary>Sentence scoring strategy.</summary>
    public SummaryMethod Method { get; init; } = SummaryMethod.Centroid;

    /// <summary>
    /// Skip a candidate whose cosine similarity to an already-selected sentence exceeds this
    /// (Maximal-Marginal-Relevance-style de-duplication). Use ≥ 1.0 to disable.
    /// </summary>
    public double RedundancyThreshold { get; init; } = 0.8;

    /// <summary>Sentences shorter than this many characters are ignored as fragments.</summary>
    public int MinSentenceChars { get; init; } = 2;

    /// <summary>
    /// Optional language code ("en", "it", "eng", "ita", …) to enable stop-word removal.
    /// When null or unknown, a script-agnostic content-word heuristic is used instead — so the
    /// summariser still works on any language with no external models (PredictiveCoding has no
    /// language detector; for the extractive task the summary stays in the source language anyway).
    /// </summary>
    public string? Language { get; init; }
}
