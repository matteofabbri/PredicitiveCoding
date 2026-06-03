using PredictiveCoding.Networks;

namespace PredictiveCoding.Expanded.Hopfield;

// ─────────────────────────────────────────────────────────────────────────────
// HopfieldNetworkConfig  —  configuration for a PCN + modern-Hopfield attractor
//
// A generative PC network learns to encode/decode data; a modern Hopfield store
// holds the latent codes of memorised exemplars.  At recall time the network's
// (possibly partial / noisy) latent is snapped to the nearest stored attractor
// before decoding, yielding pattern completion and denoising — the hippocampal
// "stabilise a coherent state" behaviour described in predictive-processing
// accounts of episodic memory.
//
// References:
//   Ramsauer et al. (2021) modern Hopfield networks.
//   Rao & Ballard (1999) — PC generative model; Friston & Buzsáki (2016).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record HopfieldNetworkConfig
{
    /// <summary>
    /// Generative PC topology.  LayerSizes[0] = data dim, LayerSizes[^1] = latent
    /// (and Hopfield) dim.
    /// </summary>
    public NetworkConfig NetworkConfig { get; init; } = new();

    /// <summary>Inverse temperature β of the Hopfield retrieval softmax.</summary>
    public float RetrievalBeta { get; init; } = 8f;

    /// <summary>Number of modern-Hopfield update iterations per recall.</summary>
    public int RetrievalIterations { get; init; } = 3;

    public string BasePath { get; init; } = "./brain/hopfield";

    public void Validate()
    {
        if (RetrievalBeta <= 0f)
            throw new InvalidOperationException("RetrievalBeta must be > 0.");
        if (RetrievalIterations < 1)
            throw new InvalidOperationException("RetrievalIterations must be >= 1.");
    }
}
