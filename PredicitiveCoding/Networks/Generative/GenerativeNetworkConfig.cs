using PredictiveCoding.Networks;

namespace PredictiveCoding.Networks.Generative;

// ─────────────────────────────────────────────────────────────────────────────
// GenerativeNetworkConfig  —  configuration for a PC-native generative model
//
// The generative network reuses a single PC hierarchy in two directions:
//
//   Encoding  (recognition):  x → inference → μ_{L-1} = z
//   Decoding  (generation):   z clamped at top → generative sweep → μ_0 = x̂
//
// A Gaussian prior N(0, 1/PriorStrength) is placed on the top layer.
// This acts like the KL term in a VAE, preventing the latent space from
// collapsing while keeping inference biologically local.
//
// References:
//   Kingma & Welling (2014) VAE — conceptual analogue without backprop.
//   Rao & Ballard (1999) — PC generative model as the theoretical basis.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record GenerativeNetworkConfig
{
    /// <summary>
    /// PC network topology.
    /// LayerSizes[0]  = input / reconstruction dimensionality.
    /// LayerSizes[^1] = latent space dimensionality.
    /// </summary>
    public NetworkConfig NetworkConfig { get; init; } = new();

    /// <summary>
    /// Precision Π_v of the isotropic Gaussian prior N(0, 1/PriorStrength)
    /// placed on the top (latent) layer.
    ///
    /// Higher values push the latent toward zero, regularising the space at
    /// the cost of reconstruction fidelity.  Set to 0 to disable the prior
    /// (maximum-likelihood inference, no KL term).
    /// </summary>
    public float PriorStrength { get; init; } = 0.1f;

    public string BasePath { get; init; } = "./brain/generative";
}
