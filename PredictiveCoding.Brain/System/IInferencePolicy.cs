namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// IInferencePolicy  —  strategy for choosing the next motor command.
//
// An inference policy is called after each observation to decide where to
// move the sensor next.  It receives the current hypothesis set and the number
// of available motor commands and returns one command index.
//
// Two ready-made policies are provided:
//   RandomPolicy      — uniform random selection (baseline)
//   UncertaintyPolicy — favours exploration when many hypotheses remain and
//                       exploitation once belief has converged
//
// Reference: Leadholm et al. (2025) P9 §4 — "action selection".
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Strategy that selects the next motor command given the current hypothesis set.</summary>
public interface IInferencePolicy
{
    /// <summary>Returns a motor command in [0, <paramref name="availableActions"/>).
    /// <paramref name="currentHypotheses"/> is the current belief set over object ids
    /// (may be empty when no object has been inferred yet).</summary>
    int SelectAction(IReadOnlyCollection<int> currentHypotheses, int availableActions);
}
