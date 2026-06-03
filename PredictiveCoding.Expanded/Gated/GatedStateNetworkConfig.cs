using PredictiveCoding.Networks;

namespace PredictiveCoding.Expanded.Gated;

// ─────────────────────────────────────────────────────────────────────────────
// GatedStateNetworkConfig  —  configuration for a PCN with a working-memory gate
//
// A working-memory cell sits on top of an ordinary PC network and is updated
// with an LSTM-style per-unit gate.  At each step the gate decides, unit by unit,
// how much of the freshly inferred latent overwrites the retained memory and how
// much of the old memory is kept:
//
//   d_i  = |z_i − m_i|                                  per-unit surprise
//   g_i  = σ( GateSlope · (d_i − GateBias) )            write gate ∈ (0, 1)
//   m_i ← (1 − g_i) · m_i + g_i · z_i                   gated update
//
// The memory is then injected as the top-layer prior (with WorkingMemoryPrecision)
// so the next inference is biased toward retained information.  Surprising units
// (large d_i) are written through; stable units are held — letting the network
// maintain relevant context across many steps for multi-step reasoning.
//
// References:
//   Hochreiter & Schmidhuber (1997) LSTM — gating analogue.
//   O'Reilly & Frank (2006) — gated working memory in PFC/basal ganglia.
//   Friston (2005) — precision-weighted priors as top-down context.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record GatedStateNetworkConfig
{
    /// <summary>
    /// PC network topology.  LayerSizes[^1] is the latent / working-memory size.
    /// </summary>
    public NetworkConfig NetworkConfig { get; init; } = new();

    /// <summary>
    /// Steepness of the write gate's sigmoid.  Larger values make the gate more
    /// decisive (closer to a hard 0/1 switch).
    /// </summary>
    public float GateSlope { get; init; } = 4f;

    /// <summary>
    /// Surprise threshold at which the write gate reaches 0.5.  Units whose change
    /// exceeds this are written through; smaller changes are mostly retained.
    /// </summary>
    public float GateBias { get; init; } = 0.5f;

    /// <summary>
    /// Precision Π_m of the working-memory prior injected on the top layer.
    /// Higher values make retained memory pull inference more strongly.
    /// </summary>
    public float WorkingMemoryPrecision { get; init; } = 1f;

    public string BasePath { get; init; } = "./brain/gated";

    public void Validate()
    {
        if (GateSlope < 0f)
            throw new InvalidOperationException("GateSlope must be >= 0.");
        if (WorkingMemoryPrecision < 0f)
            throw new InvalidOperationException("WorkingMemoryPrecision must be >= 0.");
    }
}
