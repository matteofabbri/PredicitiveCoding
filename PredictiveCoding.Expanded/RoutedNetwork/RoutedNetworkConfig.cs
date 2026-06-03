using PredictiveCoding.Networks;

namespace PredictiveCoding.Expanded.RoutedNetwork;

// ─────────────────────────────────────────────────────────────────────────────
// RoutedNetworkConfig  —  configuration for Routing-by-Prediction-Error
//
// Architecture:
//   • N expert PCNs, each with ExpertConfig topology.
//   • One router PCN that maps the raw input to a soft gate distribution
//     over the N experts.  RouterConfig.LayerSizes[^1] must equal ExpertCount.
//
// Gating logic:
//   g_i = (1 − α) · router_i + α · softmax(−β · F_i)
//
// where router_i is the router's predicted gate for expert i, F_i is the
// free energy of expert i on the current sample, β = GatingTemperature, and
// α = PeMixAlpha (0 = pure router prediction, 1 = pure PE gating ≡ MoE).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record RoutedNetworkConfig
{
    /// <summary>Shared topology for all expert PCNs.</summary>
    public NetworkConfig ExpertConfig { get; init; } = new();

    /// <summary>
    /// Router PCN topology.
    /// LayerSizes[0] must equal ExpertConfig.LayerSizes[0] (shared input).
    /// LayerSizes[^1] must equal ExpertCount (one logit per expert).
    /// </summary>
    public NetworkConfig RouterConfig { get; init; } = new();

    public int ExpertCount { get; init; } = 4;

    /// <summary>Number of experts activated (and trained) per sample.</summary>
    public int TopK { get; init; } = 2;

    /// <summary>Inverse temperature β for the PE-based softmax gate.</summary>
    public float GatingTemperature { get; init; } = 1f;

    /// <summary>
    /// Blend weight α ∈ [0, 1]:
    ///   0 = gates come entirely from the router PCN (fast, amortised).
    ///   1 = gates come entirely from per-sample PE energies (accurate, expensive).
    /// Default 0.5 balances both.
    /// </summary>
    public float PeMixAlpha { get; init; } = 0.5f;

    /// <summary>Floor applied after renormalisation to avoid zero-weight experts.</summary>
    public float MinGateWeight { get; init; } = 0.01f;

    public string BasePath { get; init; } = "./brain/routed";

    public void Validate()
    {
        if (ExpertCount < 2)
            throw new InvalidOperationException("ExpertCount must be >= 2.");
        if (TopK < 1 || TopK > ExpertCount)
            throw new InvalidOperationException($"TopK must be in [1, {ExpertCount}].");
        if (RouterConfig.LayerSizes[^1] != ExpertCount)
            throw new InvalidOperationException(
                $"RouterConfig.LayerSizes[^1] ({RouterConfig.LayerSizes[^1]}) must equal ExpertCount ({ExpertCount}).");
        if (ExpertConfig.LayerSizes[0] != RouterConfig.LayerSizes[0])
            throw new InvalidOperationException(
                "ExpertConfig and RouterConfig must share the same input dimension (LayerSizes[0]).");
        if (PeMixAlpha < 0f || PeMixAlpha > 1f)
            throw new InvalidOperationException("PeMixAlpha must be in [0, 1].");
    }
}
