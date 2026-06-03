using PredictiveCoding.Networks;

namespace PredictiveCoding.Expanded.Networks;

// ─────────────────────────────────────────────────────────────────────────────
// EnsembleConfig  —  configuration for EnsembleNetwork
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>
/// Controls how per-member class-energy distributions are aggregated at
/// prediction time.
/// </summary>
public enum EnsemblePredictionMode
{
    /// <summary>
    /// Convert each member's energy vector to a probability distribution via
    ///   p_i(c) = softmax(−β · F_i(x, one-hot(c)))
    /// then average uniformly across members and return argmax p(c).
    /// This is the standard Bayesian model averaging approach and respects
    /// uncertainty: a member that is very unsure (flat energy landscape) has
    /// less influence than one with a sharp minimum.
    /// </summary>
    ProbabilityAveraging,

    /// <summary>
    /// Average the raw energy vectors uniformly across members:
    ///   F_avg(c) = (1/N) Σ_i F_i(x, one-hot(c))
    /// then return argmin F_avg(c).
    /// Simpler and stays fully within the PC free-energy framework; works
    /// well when all members have similar energy scales.
    /// </summary>
    EnergyAveraging,
}

public sealed record EnsembleConfig
{
    /// <summary>Root directory; each member gets its own sub-folder.</summary>
    public string BasePath { get; init; } = "./ensemble";

    /// <summary>
    /// Per-member network configurations.  Members intentionally differ in
    /// depth, width, activation, or learning rates — architectural diversity
    /// is what makes an ensemble more robust than a single network.
    /// At least 2 members are required.
    /// The <c>Path</c> field on each config is overridden by
    /// <c>BasePath/member_{i}</c>.
    /// </summary>
    public NetworkConfig[] MemberConfigs { get; init; } = [];

    /// <summary>
    /// Prediction aggregation strategy.
    /// Default: <see cref="EnsemblePredictionMode.ProbabilityAveraging"/>.
    /// </summary>
    public EnsemblePredictionMode PredictionMode { get; init; } =
        EnsemblePredictionMode.ProbabilityAveraging;

    /// <summary>
    /// β used in softmax(−β · F_i) when converting energies to probabilities.
    /// Only used with <see cref="EnsemblePredictionMode.ProbabilityAveraging"/>.
    /// Higher values → sharper distributions (more winner-take-all per member).
    /// Typical range: 0.5–5.0.  Default = 1.
    /// </summary>
    public float SoftmaxTemperature { get; init; } = 1f;

    public void Validate()
    {
        if (MemberConfigs.Length < 2)
            throw new InvalidOperationException("EnsembleConfig requires at least 2 MemberConfigs.");
        if (SoftmaxTemperature <= 0f)
            throw new InvalidOperationException("SoftmaxTemperature must be > 0.");

        int outputSize = MemberConfigs[0].LayerSizes[^1];
        for (int i = 1; i < MemberConfigs.Length; i++)
        {
            if (MemberConfigs[i].LayerSizes[^1] != outputSize)
                throw new InvalidOperationException(
                    $"All members must share the same output size. " +
                    $"Member 0 has {outputSize}, member {i} has {MemberConfigs[i].LayerSizes[^1]}.");

            if (MemberConfigs[i].LayerSizes[0] != MemberConfigs[0].LayerSizes[0])
                throw new InvalidOperationException(
                    $"All members must share the same input size. " +
                    $"Member 0 has {MemberConfigs[0].LayerSizes[0]}, member {i} has {MemberConfigs[i].LayerSizes[0]}.");
        }
    }
}
