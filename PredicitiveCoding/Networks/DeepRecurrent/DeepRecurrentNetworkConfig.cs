using PredictiveCoding.Networks;

namespace PredictiveCoding.Networks.DeepRecurrent;

// ─────────────────────────────────────────────────────────────────────────────
// DeepRecurrentNetworkConfig  —  configuration for a deep recurrent PC cascade
//
// A cascade of temporal PC stages.  Stage 0 consumes the raw observation
// sequence; every higher stage consumes the latent sequence emitted by the
// stage below.  Each stage carries its own hidden-state buffer μ^{t-1} across
// timesteps, so higher stages model progressively slower / more abstract
// dynamics — the predictive-coding analogue of a deep recurrent network.
//
//   Stages[k].LayerSizes[0]  = latent dimensionality emitted by Stages[k-1]
//   Stages[k].LayerSizes[^1] = latent dimensionality emitted by Stages[k]
//
// Every stage needs at least 3 layers (input, ≥1 hidden, latent) so it can hold
// a temporal prior on its hidden layer.
//
// References:
//   Friston (2008) "Hierarchical Models in the Brain" — dynamic causal modelling.
//   Lotter, Kreiman & Cox (2017) PredNet — deep recurrent predictive coding.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record DeepRecurrentNetworkConfig
{
    /// <summary>
    /// Per-stage PC topologies, ordered bottom (fast/sensory) → top (slow/abstract).
    /// At least two stages; each stage needs ≥ 3 layers; the input dimension of
    /// each stage must equal the latent dimension of the stage below it.
    /// </summary>
    public NetworkConfig[] Stages { get; init; } = [];

    /// <summary>
    /// Per-stage temporal precision Π_t (continuity strength).  If a single value
    /// is supplied it is applied to every stage; otherwise its length must equal
    /// the number of stages.  Higher stages often use larger values to enforce
    /// slower, smoother dynamics.
    /// </summary>
    public float[] TemporalPrecision { get; init; } = [1f];

    public string BasePath { get; init; } = "./brain/deeprecurrent";

    public void Validate()
    {
        if (Stages.Length < 2)
            throw new InvalidOperationException("A DeepRecurrentNetwork needs at least 2 stages.");

        for (int k = 0; k < Stages.Length; k++)
        {
            if (Stages[k].LayerSizes.Length < 3)
                throw new InvalidOperationException(
                    $"Stage {k} needs at least 3 layers (input, ≥1 hidden, latent).");
            if (k > 0)
            {
                int below = Stages[k - 1].LayerSizes[^1];
                int above = Stages[k].LayerSizes[0];
                if (below != above)
                    throw new InvalidOperationException(
                        $"Stage {k} input ({above}) must equal stage {k - 1} latent ({below}).");
            }
        }

        if (TemporalPrecision.Length != 1 && TemporalPrecision.Length != Stages.Length)
            throw new InvalidOperationException(
                "TemporalPrecision must have length 1 (shared) or one value per stage.");
        if (TemporalPrecision.Any(p => p < 0f))
            throw new InvalidOperationException("TemporalPrecision values must be >= 0.");
    }

    /// <summary>Resolves the temporal precision for stage <paramref name="k"/>.</summary>
    public float PrecisionFor(int k) =>
        TemporalPrecision.Length == 1 ? TemporalPrecision[0] : TemporalPrecision[k];
}
