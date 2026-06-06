namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// SettlingResult  —  outcome of a full SettlingDynamics run.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record SettlingResult(
    /// <summary>Consensus from the final round.</summary>
    ConsensusResult Final,

    /// <summary>Number of rounds executed before convergence (or MaxRounds if not converged).</summary>
    int RoundsToConverge,

    /// <summary>True when <see cref="Final"/> was reached before <see cref="SettlingDynamics.MaxRounds"/>.</summary>
    bool Converged,

    /// <summary>Per-round consensus history, including the final entry.</summary>
    IReadOnlyList<ConsensusResult> History);
