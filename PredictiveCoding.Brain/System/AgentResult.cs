namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// AgentResult  —  outcome record for a single sensorimotor episode.
//
// Returned by SensorimotorAgent.RunEpisode after the agent has finished
// exploring the current object.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Outcome of a single sensorimotor recognition episode.</summary>
/// <param name="RecognizedObjectId">The id of the recognized object, or <c>null</c>
///   when the agent could not converge to a unique hypothesis within the step budget.</param>
/// <param name="StepsTaken">Number of sense-act steps executed (≥ 1, ≤ maxSteps).</param>
/// <param name="Converged">True when the agent reached a single-hypothesis belief state.</param>
/// <param name="HypothesisHistory">The hypothesis set after each step, ordered by step index.
///   Allows tracing how quickly the belief narrowed during the episode.</param>
public sealed record AgentResult(
    int? RecognizedObjectId,
    int StepsTaken,
    bool Converged,
    IReadOnlyList<IReadOnlyList<int>> HypothesisHistory);
