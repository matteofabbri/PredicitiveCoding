using PredictiveCoding.Networks;

namespace PredictiveCoding.Expanded.Networks;

// ─────────────────────────────────────────────────────────────────────────────
// TemporalMoEConfig  —  configuration for TemporalMixtureOfExperts
// ─────────────────────────────────────────────────────────────────────────────

public sealed record TemporalMoEConfig
{
    /// <summary>Root directory; each expert gets its own sub-folder.</summary>
    public string BasePath { get; init; } = "./temporal_moe";

    /// <summary>Number of temporal expert networks.</summary>
    public int ExpertCount { get; init; } = 3;

    /// <summary>
    /// Architecture and hyperparameters shared by every expert.
    /// The <c>Path</c> field is ignored — each expert is placed under
    /// <c>BasePath/expert_{i}</c>.
    /// </summary>
    public NetworkConfig ExpertConfig { get; init; } = new();

    /// <summary>
    /// Per-expert temporal precision values Π_t[i].
    /// When set, each expert enforces a different timescale of temporal
    /// smoothness — slow experts (high Π_t) track long-range structure while
    /// fast experts (low Π_t) react quickly to regime changes.
    /// Length must equal <see cref="ExpertCount"/>.
    /// When null, all experts use <see cref="DefaultTemporalPrecision"/>.
    /// </summary>
    public float[]? ExpertTemporalPrecisions { get; init; } = null;

    /// <summary>
    /// Fallback Π_t used when <see cref="ExpertTemporalPrecisions"/> is null.
    /// Typical range: 0.5–5.0.
    /// </summary>
    public float DefaultTemporalPrecision { get; init; } = 1f;

    /// <summary>
    /// β in softmax(−β · F_i): inverse temperature of the gating distribution.
    /// Higher values → harder winner-take-all.  Typical range: 0.5–5.0.
    /// </summary>
    public float GatingTemperature { get; init; } = 1f;

    /// <summary>
    /// Minimum gate weight floor after softmax to prevent expert collapse.
    /// Set to 0 to disable (pure softmax).
    /// </summary>
    public float MinGateWeight { get; init; } = 0f;

    /// <summary>Returns the temporal precision for expert <paramref name="i"/>.</summary>
    public float PrecisionFor(int i) =>
        ExpertTemporalPrecisions is not null
            ? ExpertTemporalPrecisions[i]
            : DefaultTemporalPrecision;

    public void Validate()
    {
        if (ExpertCount < 2)
            throw new InvalidOperationException("ExpertCount must be >= 2.");
        if (GatingTemperature <= 0f)
            throw new InvalidOperationException("GatingTemperature must be > 0.");
        if (MinGateWeight < 0f)
            throw new InvalidOperationException("MinGateWeight must be >= 0.");
        if (MinGateWeight >= 1f / ExpertCount)
            throw new InvalidOperationException(
                $"MinGateWeight ({MinGateWeight}) must be < 1/ExpertCount " +
                $"({1f / ExpertCount:G3}) to allow differentiation between experts.");
        if (DefaultTemporalPrecision < 0f)
            throw new InvalidOperationException("DefaultTemporalPrecision must be >= 0.");
        if (ExpertTemporalPrecisions is not null)
        {
            if (ExpertTemporalPrecisions.Length != ExpertCount)
                throw new InvalidOperationException(
                    $"ExpertTemporalPrecisions length ({ExpertTemporalPrecisions.Length}) " +
                    $"must equal ExpertCount ({ExpertCount}).");
            if (ExpertTemporalPrecisions.Any(p => p < 0f))
                throw new InvalidOperationException(
                    "All ExpertTemporalPrecisions must be >= 0.");
        }
    }
}
