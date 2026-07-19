using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Networks.Hopfield;

// ─────────────────────────────────────────────────────────────────────────────
// HopfieldNetwork  —  predictive-coding generator stabilised by an attractor
//
// Couples a generative PC network with a modern Hopfield memory over the latent
// space.  The PC network learns to encode data to latents and decode latents
// back to data; memorised exemplars are stored as latent codes in the Hopfield
// bank.  Recall from a partial or noisy input:
//
//   x̃  → Encode → z̃            (rough, possibly corrupted latent)
//   z̃  → Hopfield.Retrieve → z*  (snap to nearest stored attractor)
//   z* → Decode → x̂            (clean, completed reconstruction)
//
// The generative model supplies predictions; the attractor supplies stability,
// giving robust pattern completion under strong noise — analogous to
// hippocampal pattern completion (Friston & Buzsáki 2016).
//
// References:
//   Ramsauer et al. (2021); Rao & Ballard (1999); Friston & Buzsáki (2016).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class HopfieldNetwork : IDisposable
{
    private readonly Network               _network;
    private readonly HopfieldMemory        _memory;
    private readonly HopfieldNetworkConfig _config;

    public int InputDim   => _network.Configuration.LayerSizes[0];
    public int LatentDim  => _network.Configuration.LayerSizes[^1];
    public int MemoryCount => _memory.Count;
    public HopfieldNetworkConfig Configuration => _config;

    public HopfieldNetwork(HopfieldNetworkConfig config, Random rng)
    {
        config.Validate();
        _config = config;
        var netCfg = config.NetworkConfig with { Path = config.BasePath };
        _network = new Network(netCfg, rng);
        _memory  = new HopfieldMemory(LatentDim);
    }

    // ── Learning the generative model ───────────────────────────────────────────

    /// <summary>Unsupervised PC learning of the encode/decode generative model.</summary>
    public void Learn(DenseTensor<float> input, int inferenceSteps)
        => _network.LearnUnsupervised(input, inferenceSteps);

    // ── Memorising exemplars ────────────────────────────────────────────────────

    /// <summary>
    /// Encodes <paramref name="input"/> and stores its latent code as a Hopfield
    /// attractor.  Returns the stored latent.
    /// </summary>
    public DenseTensor<float> Store(DenseTensor<float> input, int inferenceSteps)
    {
        var z = _network.InferGetTop(input, inferenceSteps);
        _memory.Store(z);
        return z;
    }

    // ── Recall / completion ─────────────────────────────────────────────────────

    /// <summary>
    /// Retrieves the stabilised latent for <paramref name="input"/>: encode, then
    /// snap to the nearest stored Hopfield attractor.
    /// </summary>
    public DenseTensor<float> RecallLatent(DenseTensor<float> input, int inferenceSteps)
    {
        var z = _network.InferGetTop(input, inferenceSteps);
        return _memory.Retrieve(z, _config.RetrievalBeta, _config.RetrievalIterations);
    }

    /// <summary>
    /// Full pattern completion: encode a partial/noisy input, snap to the nearest
    /// attractor, and decode back to a clean reconstruction.
    /// </summary>
    public DenseTensor<float> Recall(DenseTensor<float> input, int inferenceSteps)
    {
        var zStar = RecallLatent(input, inferenceSteps);
        return _network.Generate(zStar, inferenceSteps);
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    /// <summary>Clears all stored attractors (keeps the learned generative weights).</summary>
    public void ClearMemory() => _memory.Clear();

    public float ComputeEnergy() => _network.ComputeEnergy();

    public void FlushStore() => _network.FlushStore();

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose() => _network.Dispose();
}
