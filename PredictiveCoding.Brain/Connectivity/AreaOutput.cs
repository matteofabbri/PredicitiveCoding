namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// AreaOutput  —  result of one message-passing round for a single cortical area.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record AreaOutput(
    /// <summary>Area that produced this output.</summary>
    int AreaId,

    /// <summary>Cross-column consensus after applying all incoming lateral constraints.</summary>
    ConsensusResult Consensus,

    /// <summary>Message-passing round in which this result was produced (0-based).</summary>
    int Round,

    /// <summary>Number of distinct incoming connections that contributed constraints
    /// to this area during this round.</summary>
    int IncomingMessageCount);
