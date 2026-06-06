namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// ConsensusResult  —  output of one cross-column voting round.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record ConsensusResult(
    /// <summary>Object ids that meet the agreement threshold across all columns.
    /// Empty when no object survives the vote.</summary>
    IReadOnlyList<int> Candidates,

    /// <summary>Average fraction of columns that included each surviving candidate
    /// (1.0 = unanimity, 0.5 = bare majority).</summary>
    float AgreementScore,

    /// <summary>True when at least one candidate survives the agreement threshold.</summary>
    bool IsReached,

    /// <summary>Voting round at which this result was produced (0-based).</summary>
    int Round);
