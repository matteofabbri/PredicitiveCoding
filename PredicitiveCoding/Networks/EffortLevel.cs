namespace PredictiveCoding.Networks;

// ─────────────────────────────────────────────────────────────────────────────
// EffortLevel  —  named inference-effort tiers.
//
// The LLM world exposes "reasoning effort" (low / medium / high) as an opaque
// knob; in the predictive-coding formulation the knob is simply how long the
// network is allowed to settle, and the free-energy trajectory is the native
// stop signal. A preset pairs a step budget with the early-exit threshold used
// by Network.InferAdaptive / Network.PredictAdaptive.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Named inference-effort tiers for adaptive settling.</summary>
public enum EffortLevel
{
    /// <summary>Few steps, coarse convergence — cheap answers on easy inputs.</summary>
    Low,
    /// <summary>The everyday default.</summary>
    Medium,
    /// <summary>Large step budget, tight convergence — spend compute when it matters.</summary>
    High,
}

/// <summary>
/// The single tuning point for the effort tiers: each level maps to a
/// (maxSteps, epsilon) pair — settle for at most maxSteps, stop as soon as one
/// step improves the free energy by less than epsilon RELATIVE to the current
/// energy (1e-3 = "stop below 0.1% gain per step"). Being relative, the same
/// presets are meaningful on any network size.
/// </summary>
public static class EffortPresets
{
    public static (int MaxSteps, float Epsilon) For(EffortLevel level) => level switch
    {
        EffortLevel.Low    => (10,  1e-3f),
        EffortLevel.Medium => (30,  1e-4f),
        EffortLevel.High   => (100, 1e-5f),
        _ => throw new ArgumentOutOfRangeException(nameof(level), level, "Unknown effort level."),
    };
}
