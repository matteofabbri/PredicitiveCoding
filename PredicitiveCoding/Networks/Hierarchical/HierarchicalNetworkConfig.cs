using PredictiveCoding.Networks;

namespace PredictiveCoding.Networks.Hierarchical;

// ─────────────────────────────────────────────────────────────────────────────
// HierarchicalNetworkConfig  —  configuration for a deep modular PC hierarchy
//
// A HierarchicalNetwork is a stack of autonomous PC modules.  Module 0 encodes
// the raw datum; every higher module encodes the latent state of the module
// directly below it.  Each module is itself a full generative PCN, so the stack
// behaves like a hierarchical VAE — but every level is trained by purely local
// inference (greedy, layer-wise) instead of a single global backprop pass.
//
//   Modules[k].LayerSizes[0]  = latent dimensionality emitted by Modules[k-1]
//   Modules[k].LayerSizes[^1] = latent dimensionality emitted by Modules[k]
//
// The dimensional chain must be contiguous: the input size of each module must
// equal the latent (top) size of the module beneath it.
//
// References:
//   Rao & Ballard (1999) — hierarchical predictive coding.
//   Sønderby et al. (2016) Ladder VAE — conceptual analogue with global backprop.
//   Marino (2022) "Predictive Coding, Variational Autoencoders, and …" review.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record HierarchicalNetworkConfig
{
    /// <summary>
    /// Per-module PC topologies, ordered bottom (sensory) → top (most abstract).
    /// Must contain at least two modules.  The input dimension of each module
    /// must equal the latent (top) dimension of the module below it.
    /// </summary>
    public NetworkConfig[] Modules { get; init; } = [];

    public string BasePath { get; init; } = "./brain/hierarchical";

    public void Validate()
    {
        if (Modules.Length < 2)
            throw new InvalidOperationException("A HierarchicalNetwork needs at least 2 modules.");

        for (int k = 1; k < Modules.Length; k++)
        {
            int below = Modules[k - 1].LayerSizes[^1];
            int above = Modules[k].LayerSizes[0];
            if (below != above)
                throw new InvalidOperationException(
                    $"Module {k} input ({above}) must equal module {k - 1} latent ({below}).");
        }
    }
}
