using PredictiveCoding.Core;
using PredictiveCoding.Networks.MoE;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// PremotorCortex (SMA)  —  action sequencing / chunking
//
// The premotor cortex and supplementary motor area organise individual movements
// into ordered sequences, predicting the next action given the current state and the
// learned temporal context.  Implemented over a TemporalMixtureOfExperts whose
// experts specialise on different temporal regimes of the action sequence; it feeds
// the chosen action to MotorCortex for execution.
//
// References: Tanji (2001) sequential movement representation; Friston (2008).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record PremotorCortexConfig
{
    /// <summary>Temporal MoE over the action sequence.  ExpertConfig.LayerSizes[0] = state dim,
    /// LayerSizes[^1] = number of actions.</summary>
    public TemporalMoEConfig Sequencer { get; init; } = new();

    public string BasePath { get; init; } = "./cortex/premotor";

    public void Validate() => Sequencer.Validate();
}

public sealed class PremotorCortex : IDisposable
{
    private readonly TemporalMixtureOfExperts _seq;

    public int ExpertCount => _seq.ExpertCount;

    public PremotorCortex(PremotorCortexConfig config, Random rng)
    {
        config.Validate();
        _seq = new TemporalMixtureOfExperts(config.Sequencer with { BasePath = config.BasePath }, rng);
    }

    /// <summary>Starts a new action sequence (clears temporal context).</summary>
    public void Reset() => _seq.Reset();

    /// <summary>Learns one step of a motor sequence: current state → next action (one-hot).</summary>
    public void LearnStep(DenseTensor<float> state, DenseTensor<float> action, int steps) => _seq.Step(state, action, steps);

    /// <summary>Predicts the next action index in the sequence given the current state.</summary>
    public int NextAction(DenseTensor<float> state, int steps) => _seq.Predict(state, steps);

    public void FlushAll() => _seq.FlushAll();
    public void Dispose()  => _seq.Dispose();
}
