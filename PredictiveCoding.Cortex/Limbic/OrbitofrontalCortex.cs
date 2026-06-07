using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// OrbitofrontalCortex  —  expected-value / outcome model
//
// The OFC represents the expected value of stimuli and outcomes and updates it as
// contingencies change, supplying the value estimate that the basal ganglia use to
// select actions.  Implemented as a PCN mapping a stimulus latent to a scalar
// expected value, learned from observed outcomes.
//
// References: Rolls (2000); Rushworth et al. (2011) reward-guided decision-making.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record OrbitofrontalCortexConfig
{
    /// <summary>Value PCN: stimulus → scalar expected value.  LayerSizes[^1] must be 1.</summary>
    public NetworkConfig Value { get; init; } = new();

    public string BasePath { get; init; } = "./cortex/ofc";

    public void Validate()
    {
        if (Value.LayerSizes[^1] != 1)
            throw new InvalidOperationException(
                $"Value.LayerSizes[^1] ({Value.LayerSizes[^1]}) must be 1 (scalar expected value).");
    }
}

public sealed class OrbitofrontalCortex : IDisposable
{
    private readonly Network _value;

    public int StimulusDim => _value.Configuration.LayerSizes[0];

    public OrbitofrontalCortex(OrbitofrontalCortexConfig config, Random rng)
    {
        config.Validate();
        _value = new Network(config.Value with { Path = config.BasePath }, rng);
    }

    /// <summary>Expected value (reward) of a stimulus / outcome.</summary>
    public float EstimateValue(DenseTensor<float> stimulus, int steps)
        => _value.InferGetTop(stimulus, steps).Buffer[0];

    /// <summary>Learns the stimulus → value association from an observed outcome.</summary>
    public void Learn(DenseTensor<float> stimulus, float value, int steps)
        => _value.Learn(stimulus, new DenseTensor<float>([value], 1), steps);

    public void FlushAll() => _value.FlushStore();
    public void Dispose()  => _value.Dispose();
}
