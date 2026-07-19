using PredictiveCoding.Networks;

namespace PredictiveCoding.Networks.Attentive;

// ─────────────────────────────────────────────────────────────────────────────
// AttentiveNetworkConfig  —  configuration for AttentiveNetwork
// ─────────────────────────────────────────────────────────────────────────────

public sealed record AttentiveNetworkConfig
{
    /// <summary>Base network configuration (layer sizes, activation, learning rates, etc.).</summary>
    public NetworkConfig BaseConfig { get; init; } = new();

    /// <summary>
    /// η_att: learning rate for the per-unit attention logit update.
    /// Should be one or two orders of magnitude smaller than
    /// <see cref="NetworkConfig.WeightLearningRate"/> — attention specialises
    /// slowly, over many examples.  Typical range: 0.001–0.05.
    /// </summary>
    public float AttentionLearningRate { get; init; } = 0.01f;

    /// <summary>
    /// λ_att: L2 regularisation coefficient applied to the raw logits s_l.
    /// Pulls the logit toward 0, which corresponds to α_l[i] = 0.5 (neutral
    /// attention, neither suppressing nor amplifying the error).
    /// Prevents logits from drifting to ±∞ and controls how "peaked" the
    /// final attention profile becomes.  Typical range: 0.01–0.5.
    /// </summary>
    public float AttentionRegularization { get; init; } = 0.1f;

    public void Validate()
    {
        if (AttentionLearningRate < 0f)
            throw new InvalidOperationException("AttentionLearningRate must be >= 0.");
        if (AttentionRegularization < 0f)
            throw new InvalidOperationException("AttentionRegularization must be >= 0.");
        BaseConfig.Validate();
    }
}
