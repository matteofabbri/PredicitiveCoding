using PredictiveCoding.Core;

namespace PredictiveCoding.Expanded.Spiking;

// ─────────────────────────────────────────────────────────────────────────────
// SpikingNetworkConfig  —  configuration for a spiking predictive-coding network
//
// A neuromorphic realisation of hierarchical predictive coding.  Each layer is a
// population of leaky integrate-and-fire (LIF) neurons; their low-pass-filtered
// firing rate r_l carries the layer state, and prediction errors are encoded as
// rate differences ε_{l-1} = r_{l-1} − f(W_l r_l + b_l).  Membrane potentials are
// driven by the bottom-up error and the top-down precision-weighted error, so the
// network minimises the same free energy as the rate model but with spikes —
// suitable for neuromorphic hardware and energy-efficient simulation.
//
// References:
//   Boerlin, Machens & Denève (2013) — predictive spike coding.
//   Rao & Ballard (1999) — hierarchical predictive coding.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record SpikingNetworkConfig
{
    /// <summary>Layer sizes input → output, e.g. [64, 32, 10].</summary>
    public int[] LayerSizes { get; init; } = [20, 32, 4];

    /// <summary>Non-linearity of the generative prediction f(W r + b).</summary>
    public ActivationType Activation { get; init; } = ActivationType.Tanh;

    /// <summary>Number of simulation ticks per inference (spiking time horizon).</summary>
    public int Ticks { get; init; } = 32;

    /// <summary>Integration step Δt for the membrane dynamics.</summary>
    public float Dt { get; init; } = 1f;

    /// <summary>Membrane leak λ: pulls the potential back toward rest each tick.</summary>
    public float Leak { get; init; } = 0.1f;

    /// <summary>Drive gain η_V: how strongly prediction errors charge the membrane.</summary>
    public float DriveRate { get; init; } = 0.2f;

    /// <summary>Firing threshold ϑ; a neuron spikes and resets when V ≥ ϑ.</summary>
    public float Threshold { get; init; } = 1f;

    /// <summary>
    /// Rate low-pass coefficient ρ ∈ [0, 1): r ← ρ·r + (1−ρ)·spike.
    /// Higher ρ gives smoother (slower) rate estimates.
    /// </summary>
    public float RateDecay { get; init; } = 0.5f;

    /// <summary>Hebbian weight learning rate η_W.</summary>
    public float WeightLearningRate { get; init; } = 0.01f;

    public void Validate()
    {
        if (LayerSizes.Length < 2)
            throw new InvalidOperationException("LayerSizes must have at least 2 elements.");
        if (LayerSizes.Any(s => s < 1))
            throw new InvalidOperationException("All LayerSizes must be >= 1.");
        if (Ticks < 1) throw new InvalidOperationException("Ticks must be >= 1.");
        if (Dt <= 0f) throw new InvalidOperationException("Dt must be > 0.");
        if (Leak < 0f) throw new InvalidOperationException("Leak must be >= 0.");
        if (DriveRate < 0f) throw new InvalidOperationException("DriveRate must be >= 0.");
        if (Threshold <= 0f) throw new InvalidOperationException("Threshold must be > 0.");
        if (RateDecay < 0f || RateDecay >= 1f)
            throw new InvalidOperationException("RateDecay must be in [0, 1).");
        if (WeightLearningRate < 0f)
            throw new InvalidOperationException("WeightLearningRate must be >= 0.");
    }
}
