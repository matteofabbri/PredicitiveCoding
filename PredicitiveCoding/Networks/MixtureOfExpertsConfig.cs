namespace PredictiveCoding.Networks;

// ─────────────────────────────────────────────────────────────────────────────
// MixtureOfExpertsConfig
// ─────────────────────────────────────────────────────────────────────────────

public sealed record MixtureOfExpertsConfig
{
    /// <summary>Root directory; each expert gets its own sub-folder.</summary>
    public string BasePath { get; init; } = "./moe";

    /// <summary>Number of expert networks.</summary>
    public int ExpertCount { get; init; } = 3;

    /// <summary>
    /// Architecture and hyperparameters shared by every expert.
    /// The <c>Path</c> field on this config is ignored — each expert is placed
    /// under <c>BasePath/expert_{i}</c>.
    /// </summary>
    public NetworkConfig ExpertConfig { get; init; } = new();

    /// <summary>
    /// β in softmax(−β · F_i): inverse temperature of the gating distribution.
    /// Higher values → harder (more winner-take-all) assignment.
    /// Lower values  → softer (more uniform) assignment.
    /// Typical range: 0.5–5.0.  Default = 1.
    /// </summary>
    public float GatingTemperature { get; init; } = 1f;

    /// <summary>
    /// Minimum gate weight floor applied after softmax to prevent expert
    /// collapse (any expert falling permanently to zero activity).
    /// Set to 0 to disable the floor (pure softmax).
    /// Default = 0.
    /// </summary>
    public float MinGateWeight { get; init; } = 0f;

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
                $"MinGateWeight ({MinGateWeight}) must be < 1/ExpertCount ({1f / ExpertCount:G3}) " +
                "to allow any differentiation between experts.");
    }
}
