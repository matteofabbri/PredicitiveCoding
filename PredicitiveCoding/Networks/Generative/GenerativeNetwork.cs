using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Networks.Generative;

// ─────────────────────────────────────────────────────────────────────────────
// GenerativeNetwork  —  biologically-plausible VAE without backprop
//
// A single PC hierarchy is used in two directions:
//
//   Encode  (recognition):
//     x clamped at layer[0] → iterative inference → μ_{L-1} = z
//
//   Decode  (generation):
//     z clamped at layer[L-1] → generative sweep → μ_0 = x̂
//
// Learning is unsupervised: only the bottom layer is clamped to the datum;
// the top layer runs free inference regularised by a Gaussian prior N(0, I/Π_v).
// Weight updates follow the standard Hebbian PC rule (no global backprop).
//
// At inference the same weights that generate (top-down) also constrain the
// encoding (bottom-up), so the two directions are automatically consistent.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class GenerativeNetwork : IDisposable
{
    private readonly Network                 _network;
    private readonly GenerativeNetworkConfig _config;

    /// <summary>Dimensionality of the input / reconstruction space (layer[0]).</summary>
    public int InputDim  { get; }

    /// <summary>Dimensionality of the latent space (layer[L-1]).</summary>
    public int LatentDim { get; }

    public GenerativeNetworkConfig Configuration => _config;

    public GenerativeNetwork(GenerativeNetworkConfig config, Random rng)
    {
        _config = config;
        var netCfg = config.NetworkConfig with { Path = config.BasePath };
        _network   = new Network(netCfg, rng);

        InputDim  = netCfg.LayerSizes[0];
        LatentDim = netCfg.LayerSizes[^1];

        // Install a zero-mean Gaussian prior on the top layer (KL term analogue).
        if (config.PriorStrength > 0f)
        {
            var prior = DenseTensor<float>.Zeros(LatentDim);
            _network.SetTopLayerPrior(prior, config.PriorStrength);
        }
    }

    // ── Encoding ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Encodes <paramref name="input"/> to a latent representation by running
    /// free inference (top layer unclamped) and returning μ_{L-1}.
    /// </summary>
    public DenseTensor<float> Encode(DenseTensor<float> input, int inferenceSteps)
        => _network.InferGetTop(input, inferenceSteps);

    // ── Decoding ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Decodes <paramref name="latent"/> back to the input space by clamping
    /// the top layer and running the generative inference loop.
    /// Returns layer[0].Mu as the reconstructed sample.
    /// </summary>
    public DenseTensor<float> Decode(DenseTensor<float> latent, int inferenceSteps)
        => _network.Generate(latent, inferenceSteps);

    // ── Reconstruction ───────────────────────────────────────────────────────

    /// <summary>Full auto-encoding round-trip: x → z → x̂.</summary>
    public DenseTensor<float> Reconstruct(DenseTensor<float> input, int inferenceSteps)
    {
        var z = Encode(input, inferenceSteps);
        return Decode(z, inferenceSteps);
    }

    /// <summary>
    /// Half mean-squared reconstruction error:  ½ ||x − Decode(Encode(x))||².
    /// Lower is better; 0 means perfect reconstruction.
    /// </summary>
    public float ReconstructionError(DenseTensor<float> input, int inferenceSteps)
    {
        var flat  = input.Rank == 1 ? input : TensorUtils.Reshape(input, InputDim);
        var x_hat = Reconstruct(flat, inferenceSteps);
        return 0.5f * TensorUtils.SumSquares(TensorUtils.ElementwiseSub(flat, x_hat));
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Unsupervised learning step:
    ///   1. Clamp layer[0] to <paramref name="input"/>.
    ///   2. Run free inference — top layer is guided only by the prior.
    ///   3. Update all generative weights via the Hebbian PC rule.
    /// </summary>
    public void Learn(DenseTensor<float> input, int inferenceSteps)
        => _network.LearnUnsupervised(input, inferenceSteps);

    // ── Sampling ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Samples z ~ N(0, I) and decodes it to a novel generated sample.
    /// </summary>
    public DenseTensor<float> Sample(int inferenceSteps, Random rng)
    {
        var z = RandomUtils.SampleNormal([LatentDim], 0f, 1f, rng);
        return Decode(z, inferenceSteps);
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    /// <summary>Global PC free energy after the last inference pass.</summary>
    public float ComputeEnergy() => _network.ComputeEnergy();

    public void FlushStore() => _network.FlushStore();

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose() => _network.Dispose();
}
