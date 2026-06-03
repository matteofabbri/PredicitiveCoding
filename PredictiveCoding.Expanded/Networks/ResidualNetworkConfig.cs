using PredictiveCoding.Networks;
using PredictiveCoding.Core;

namespace PredictiveCoding.Expanded.Networks;

// ─────────────────────────────────────────────────────────────────────────────
// ResidualNetworkConfig  —  configuration for ResidualNetwork
// ─────────────────────────────────────────────────────────────────────────────

public sealed record ResidualNetworkConfig
{
    /// <summary>
    /// Base network configuration (layer sizes, activation, learning rates, etc.).
    /// Requires at least 3 layers so that at least one skip connection exists.
    /// </summary>
    public NetworkConfig BaseConfig { get; init; } = new();

    /// <summary>
    /// π_skip: precision (weight) of skip-connection top-down predictions.
    /// Scales how strongly the 2-hop prediction contributes to the combined
    /// prediction error at the target layer:
    ///   combined_pred_l = adj_pred_l + π_skip · skip_pred_l
    /// Typical range: 0.1–1.0.  Default = 0.5.
    /// Set to 0 to disable skip contributions (equivalent to a standard network).
    /// </summary>
    public float SkipPrecision { get; init; } = 0.5f;

    /// <summary>
    /// Learning rate for skip-connection weight matrices V and biases c.
    /// When null, uses <see cref="NetworkConfig.WeightLearningRate"/> from
    /// <see cref="BaseConfig"/>.
    /// </summary>
    public float? SkipWeightLearningRate { get; init; } = null;

    /// <summary>
    /// Activation function applied to skip-connection pre-activations.
    /// When null, uses <see cref="NetworkConfig.Activation"/> from
    /// <see cref="BaseConfig"/>.
    /// </summary>
    public ActivationType? SkipActivation { get; init; } = null;

    // ── Resolved helpers (internal) ───────────────────────────────────────────

    internal float EffectiveSkipLR =>
        SkipWeightLearningRate ?? BaseConfig.WeightLearningRate;

    internal ActivationType EffectiveSkipActivation =>
        SkipActivation ?? BaseConfig.Activation;

    // ── Validation ────────────────────────────────────────────────────────────

    public void Validate()
    {
        if (SkipPrecision < 0f)
            throw new InvalidOperationException("SkipPrecision must be >= 0.");
        if (SkipWeightLearningRate.HasValue && SkipWeightLearningRate.Value < 0f)
            throw new InvalidOperationException("SkipWeightLearningRate must be >= 0.");
        if (BaseConfig.LayerSizes.Length < 3)
            throw new InvalidOperationException(
                "ResidualNetwork requires at least 3 layers (input, ≥1 hidden, output) " +
                "to form at least one skip connection.");
        BaseConfig.Validate();
    }
}
