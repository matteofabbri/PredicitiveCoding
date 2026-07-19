using PredictiveCoding.Core;
using PredictiveCoding.Networks.DeepRecurrent;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// InsularCortex  —  interoception (body-state → feeling)
//
// The insula integrates the body's internal signals (heart rate, temperature,
// hunger, …) over time into a representation of the current bodily/feeling state —
// the substrate of interoceptive awareness.  Implemented over a DeepRecurrentNetwork
// that integrates the interoceptive stream into a feeling latent.
//
// References: Craig (2002) "How do you feel?"; Seth (2013) interoceptive inference.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record InsularCortexConfig
{
    /// <summary>Temporal interoceptive integrator.  Stages[0].LayerSizes[0] = body-state dim.</summary>
    public DeepRecurrentNetworkConfig Interoceptive { get; init; } = new();

    public string BasePath { get; init; } = "./cortex/insula";

    public void Validate() => Interoceptive.Validate();
}

public sealed class InsularCortex : IDisposable
{
    private readonly DeepRecurrentNetwork _stream;

    public InsularCortex(InsularCortexConfig config, Random rng)
    {
        config.Validate();
        _stream = new DeepRecurrentNetwork(config.Interoceptive with { BasePath = config.BasePath }, rng);
    }

    /// <summary>Starts a new interoceptive episode (clears temporal context).</summary>
    public void Reset() => _stream.Reset();

    /// <summary>Integrates one body-state sample over time (learning); returns the feeling latent.</summary>
    public DenseTensor<float> Experience(DenseTensor<float> bodyState, int steps) => _stream.Step(bodyState, steps);

    /// <summary>Reads the current feeling latent for a body-state (inference, no learning).</summary>
    public DenseTensor<float> Feel(DenseTensor<float> bodyState, int steps) => _stream.Encode(bodyState, steps);

    public void FlushAll() => _stream.FlushAll();
    public void Dispose()  => _stream.Dispose();
}
