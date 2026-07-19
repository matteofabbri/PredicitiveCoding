using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Networks.DeepRecurrent;

// ─────────────────────────────────────────────────────────────────────────────
// DeepRecurrentNetwork  —  cascade of temporal PCNs (deep recurrent PC)
//
// A stack of temporal PC stages.  At each timestep the raw observation is fed
// into stage 0; the latent it emits becomes the input to stage 1, and so on up
// the cascade.  Every stage keeps its own hidden-state buffer μ^{t-1} and injects
// it as a temporal prior before inference, so each level models the dynamics of
// the level beneath it rather than a single static map.  Higher stages therefore
// capture slower, more abstract temporal structure — the PC counterpart of a
// deep recurrent network, but trained entirely by local Hebbian updates.
//
// Per timestep:
//   cur = x_t
//   for k = 0 … N-1:
//       (z_k, h_k) = Stage[k].LearnUnsupervisedTemporal(cur, steps, h_k, Π_k)
//       cur = z_k                         // latent feeds the stage above
//
// References:
//   Friston (2008) — dynamic causal modelling / hierarchical dynamics.
//   Lotter, Kreiman & Cox (2017) PredNet — deep recurrent predictive coding.
//   Millidge, Tschantz & Buckley (2021) — predictive coding review.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class DeepRecurrentNetwork : IDisposable
{
    private readonly Network[]                   _stages;
    private readonly DeepRecurrentNetworkConfig  _config;
    private readonly DenseTensor<float>[]?[]     _priors;   // per-stage hidden states μ^{t-1}

    /// <summary>Number of cascaded temporal stages.</summary>
    public int StageCount => _stages.Length;

    /// <summary>Dimensionality of the raw input sequence (stage 0, layer[0]).</summary>
    public int InputDim => _stages[0].Configuration.LayerSizes[0];

    /// <summary>Dimensionality of the top-stage latent (slowest dynamics).</summary>
    public int TopLatentDim => _stages[^1].Configuration.LayerSizes[^1];

    /// <summary>True once at least one timestep has been processed since the last Reset.</summary>
    public bool HasContext => _priors[0] != null;

    public DeepRecurrentNetworkConfig Configuration => _config;

    public DeepRecurrentNetwork(DeepRecurrentNetworkConfig config, Random rng)
    {
        config.Validate();
        _config = config;

        _stages = new Network[config.Stages.Length];
        for (int k = 0; k < config.Stages.Length; k++)
        {
            var netCfg = config.Stages[k] with
            {
                Path = Path.Combine(config.BasePath, $"stage_{k}"),
            };
            _stages[k] = new Network(netCfg, new Random(rng.Next()));
        }

        _priors = new DenseTensor<float>[]?[config.Stages.Length];
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Processes one timestep of a sequence in learning mode.  Each stage injects
    /// its previous hidden state as a temporal prior, runs unsupervised inference,
    /// updates its weights, and emits a latent that feeds the stage above.
    /// Returns the top-stage latent z_{N-1} for this timestep.
    /// </summary>
    public DenseTensor<float> Step(DenseTensor<float> input, int inferenceSteps)
    {
        var cur = input;
        for (int k = 0; k < _stages.Length; k++)
        {
            var (z, hidden) = _stages[k].LearnUnsupervisedTemporal(
                cur, inferenceSteps, _priors[k], _config.PrecisionFor(k));
            _priors[k] = hidden;
            cur        = z;
        }
        return cur;
    }

    // ── Inference / encoding ───────────────────────────────────────────────────

    /// <summary>
    /// Processes one timestep WITHOUT updating weights, rolling the temporal
    /// context forward.  Returns the top-stage latent.  Use during evaluation to
    /// keep the cascade's dynamics in sync with the observed sequence.
    /// </summary>
    public DenseTensor<float> Encode(DenseTensor<float> input, int inferenceSteps)
    {
        var cur = input;
        for (int k = 0; k < _stages.Length; k++)
        {
            var (z, hidden) = _stages[k].InferGetTopTemporal(
                cur, inferenceSteps, _priors[k], _config.PrecisionFor(k));
            _priors[k] = hidden;
            cur        = z;
        }
        return cur;
    }

    // ── Sequence control ───────────────────────────────────────────────────────

    /// <summary>
    /// Clears every stage's hidden-state buffer.  Call at the start of each new
    /// independent sequence to prevent cross-sequence contamination.
    /// </summary>
    public void Reset()
    {
        for (int k = 0; k < _priors.Length; k++) _priors[k] = null;
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    /// <summary>PC free energy of stage <paramref name="stageIndex"/> after its last inference.</summary>
    public float StageEnergy(int stageIndex) => _stages[stageIndex].ComputeEnergy();

    public void FlushAll()
    {
        foreach (var s in _stages) s.FlushStore();
    }

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose()
    {
        foreach (var s in _stages) s.Dispose();
    }
}
