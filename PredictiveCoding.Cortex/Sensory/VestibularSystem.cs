using PredictiveCoding.Core;
using PredictiveCoding.Expanded.DeepRecurrent;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// VestibularSystem  —  self-motion integration / head direction
//
// The vestibular system reports head acceleration and angular velocity; downstream
// head-direction cells integrate that motion over time into a persistent estimate
// of heading.  Implemented over a DeepRecurrentNetwork that integrates a stream of
// motion vectors into a head-direction code, the temporal partner of the entorhinal
// grid-cell code for path integration.
//
// References: Taube (2007) head-direction signal; Cullen (2012) vestibular system.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record VestibularSystemConfig
{
    /// <summary>Temporal motion integrator.  Stages[0].LayerSizes[0] = motion-vector dim;
    /// Stages[^1].LayerSizes[^1] = head-direction code dim.</summary>
    public DeepRecurrentNetworkConfig Integrator { get; init; } = new();

    public string BasePath { get; init; } = "./cortex/vestibular";

    public void Validate() => Integrator.Validate();
}

public sealed class VestibularSystem : IDisposable
{
    private readonly DeepRecurrentNetwork _integrator;

    public int MotionDim        => _integrator.InputDim;
    public int HeadDirectionDim => _integrator.TopLatentDim;

    public VestibularSystem(VestibularSystemConfig config, Random rng)
    {
        config.Validate();
        _integrator = new DeepRecurrentNetwork(config.Integrator with { BasePath = config.BasePath }, rng);
    }

    /// <summary>Starts a new motion episode (clears the heading estimate).</summary>
    public void Reset() => _integrator.Reset();

    /// <summary>Integrates one motion sample over time (learning); returns the head-direction code.</summary>
    public DenseTensor<float> Integrate(DenseTensor<float> motion, int steps) => _integrator.Step(motion, steps);

    /// <summary>Reads the current head-direction code for a motion sample (inference, no learning).</summary>
    public DenseTensor<float> HeadDirection(DenseTensor<float> motion, int steps) => _integrator.Encode(motion, steps);

    public void FlushAll() => _integrator.FlushAll();
    public void Dispose()  => _integrator.Dispose();
}
