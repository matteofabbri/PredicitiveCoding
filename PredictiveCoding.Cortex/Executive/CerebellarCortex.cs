using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// CerebellarCortex  —  fast supervised forward model
//
// The cerebellum is classically a forward model: given the current state and a
// motor command it predicts the sensory consequence, and learns from the error
// between prediction and actual outcome.  Unlike the MotorCortex agent it does
// not select actions — it is a pure, fast predictor used for online correction
// and timing.  Implemented as a PCN mapping [state ⊕ command] → next state.
//
// References: Wolpert, Miall & Kawato (1998) cerebellar forward models;
//             Friston (2008).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record CerebellarCortexConfig
{
    public int StateDim   { get; init; }
    public int CommandDim { get; init; }

    /// <summary>Forward-model PCN.  LayerSizes[0] = StateDim + CommandDim,
    /// LayerSizes[^1] = StateDim.</summary>
    public NetworkConfig ForwardModel { get; init; } = new();

    public string BasePath { get; init; } = "./cortex/cerebellum";

    public void Validate()
    {
        if (StateDim < 1)   throw new InvalidOperationException("StateDim must be >= 1.");
        if (CommandDim < 1) throw new InvalidOperationException("CommandDim must be >= 1.");
        if (ForwardModel.LayerSizes[0] != StateDim + CommandDim)
            throw new InvalidOperationException(
                $"ForwardModel.LayerSizes[0] ({ForwardModel.LayerSizes[0]}) must equal StateDim + CommandDim ({StateDim + CommandDim}).");
        if (ForwardModel.LayerSizes[^1] != StateDim)
            throw new InvalidOperationException(
                $"ForwardModel.LayerSizes[^1] ({ForwardModel.LayerSizes[^1]}) must equal StateDim ({StateDim}).");
    }
}

public sealed class CerebellarCortex : IDisposable
{
    private readonly Network               _forward;
    private readonly CerebellarCortexConfig _config;

    public int StateDim   => _config.StateDim;
    public int CommandDim => _config.CommandDim;

    public CerebellarCortex(CerebellarCortexConfig config, Random rng)
    {
        config.Validate();
        _config  = config;
        _forward = new Network(config.ForwardModel with { Path = config.BasePath }, rng);
    }

    /// <summary>Predicts the next state produced by applying <paramref name="command"/> in <paramref name="state"/>.</summary>
    public DenseTensor<float> Predict(DenseTensor<float> state, DenseTensor<float> command, int steps)
        => _forward.InferGetTop(BuildInput(state, command), steps);

    /// <summary>Learns the transition (state, command) → actualNext from observed outcome.</summary>
    public void Learn(DenseTensor<float> state, DenseTensor<float> command, DenseTensor<float> actualNext, int steps)
    {
        if (actualNext.Length != _config.StateDim)
            throw new ArgumentException($"actualNext has {actualNext.Length} elements; expected StateDim ({_config.StateDim}).");
        _forward.Learn(BuildInput(state, command), actualNext, steps);
    }

    /// <summary>Half mean-squared forward-model error ½‖predicted − actual‖².</summary>
    public float PredictionError(DenseTensor<float> state, DenseTensor<float> command, DenseTensor<float> actualNext, int steps)
    {
        var pred = Predict(state, command, steps);
        return 0.5f * TensorUtils.SumSquares(TensorUtils.ElementwiseSub(pred, actualNext));
    }

    public void FlushAll() => _forward.FlushStore();
    public void Dispose()  => _forward.Dispose();

    private DenseTensor<float> BuildInput(DenseTensor<float> state, DenseTensor<float> command)
    {
        if (state.Length != _config.StateDim)
            throw new ArgumentException($"state has {state.Length} elements; expected StateDim ({_config.StateDim}).");
        if (command.Length != _config.CommandDim)
            throw new ArgumentException($"command has {command.Length} elements; expected CommandDim ({_config.CommandDim}).");

        var input = new DenseTensor<float>(_config.StateDim + _config.CommandDim);
        Array.Copy(state.Buffer,   0, input.Buffer, 0,               _config.StateDim);
        Array.Copy(command.Buffer, 0, input.Buffer, _config.StateDim, _config.CommandDim);
        return input;
    }
}
