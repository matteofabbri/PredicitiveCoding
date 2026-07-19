using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Networks.Active;

// ─────────────────────────────────────────────────────────────────────────────
// ActiveNetwork  —  Active Predictive-Coding agent (perception–action loop)
//
// A generative world model (PCN) predicts the next observation from the current
// observation and a candidate action.  A policy then selects the action that
// minimises expected free energy G(a):
//
//   ŷ(a)  = WorldModel( [obs ⊕ one-hot(a)] )           predicted next observation
//   G(a)  = λ_goal · ½‖ŷ(a) − goal‖²                   pragmatic value (reach goal)
//           − λ_explore · ½‖ŷ(a) − obs‖²               epistemic value (seek novelty)
//   a*    = argmin_a G(a)
//
// Learning is ordinary supervised PC: the world model clamps its top layer to the
// observed next state and updates weights via the local Hebbian rule.  No global
// backprop and no separate critic network are required — the same generative
// model that perceives is the one that plans.
//
// References:
//   Friston et al. (2017) "Active Inference: A Process Theory", Neural Comput.
//   Rao, Gklezakos, Sathish (2024) "Active Predictive Coding", Neural Comput.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class ActiveNetwork : IDisposable
{
    private readonly Network            _worldModel;
    private readonly ActiveNetworkConfig _config;

    public int ObservationDim => _config.ObservationDim;
    public int ActionCount    => _config.ActionCount;
    public ActiveNetworkConfig Configuration => _config;

    public ActiveNetwork(ActiveNetworkConfig config, Random rng)
    {
        config.Validate();
        _config = config;
        var netCfg = config.WorldModelConfig with { Path = config.BasePath };
        _worldModel = new Network(netCfg, rng);
    }

    // ── World-model prediction ────────────────────────────────────────────────

    /// <summary>
    /// Predicts the next observation that would follow taking
    /// <paramref name="action"/> in <paramref name="observation"/> by clamping the
    /// world model's input to [obs ⊕ one-hot(action)] and reading the inferred
    /// top layer.  Transient — no weights are modified.
    /// </summary>
    public DenseTensor<float> PredictNext(
        DenseTensor<float> observation, int action, int inferenceSteps)
    {
        var input = BuildInput(observation, action);
        return _worldModel.InferGetTop(input, inferenceSteps);
    }

    // ── Action selection ──────────────────────────────────────────────────────

    /// <summary>
    /// Selects the action minimising expected free energy toward
    /// <paramref name="goal"/>, and returns it together with the per-action scores
    /// (lower = better).  Pragmatic term pulls the predicted outcome toward the
    /// goal; the optional epistemic term rewards outcomes far from the current
    /// observation (exploration).
    /// </summary>
    public (int action, float[] expectedFreeEnergy) SelectAction(
        DenseTensor<float> observation, DenseTensor<float> goal, int inferenceSteps)
    {
        if (goal.Length != _config.ObservationDim)
            throw new ArgumentException(
                $"goal has {goal.Length} elements; expected ObservationDim ({_config.ObservationDim}).");

        var   scores    = new float[_config.ActionCount];
        int   bestA     = 0;
        float bestScore = float.MaxValue;

        for (int a = 0; a < _config.ActionCount; a++)
        {
            var predicted  = PredictNext(observation, a, inferenceSteps);
            float pragmatic = 0.5f * TensorUtils.SumSquares(TensorUtils.ElementwiseSub(predicted, goal));
            float epistemic = _config.ExplorationWeight > 0f
                ? 0.5f * TensorUtils.SumSquares(TensorUtils.ElementwiseSub(predicted, observation))
                : 0f;

            float g  = _config.GoalPrecision * pragmatic - _config.ExplorationWeight * epistemic;
            scores[a] = g;
            if (g < bestScore) { bestScore = g; bestA = a; }
        }

        return (bestA, scores);
    }

    /// <summary>Convenience wrapper returning only the chosen action.</summary>
    public int Act(DenseTensor<float> observation, DenseTensor<float> goal, int inferenceSteps)
        => SelectAction(observation, goal, inferenceSteps).action;

    // ── Learning ──────────────────────────────────────────────────────────────

    /// <summary>
    /// Learns one transition (obs, action) → nextObs by clamping the world model's
    /// top layer to the observed next state and updating weights with the Hebbian
    /// PC rule.  Call this after every environment step to refine the model.
    /// </summary>
    public void Observe(
        DenseTensor<float> observation,
        int                action,
        DenseTensor<float> nextObservation,
        int                inferenceSteps)
    {
        if (nextObservation.Length != _config.ObservationDim)
            throw new ArgumentException(
                $"nextObservation has {nextObservation.Length} elements; " +
                $"expected ObservationDim ({_config.ObservationDim}).");

        var input = BuildInput(observation, action);
        _worldModel.Learn(input, nextObservation, inferenceSteps);
    }

    /// <summary>
    /// One full closed-loop step in a known environment: select the best action
    /// toward <paramref name="goal"/>, then learn from the actual outcome obtained
    /// by applying <paramref name="environment"/>.  Returns the chosen action and
    /// the realised next observation.
    /// </summary>
    public (int action, DenseTensor<float> nextObservation) Step(
        DenseTensor<float>                          observation,
        DenseTensor<float>                          goal,
        Func<DenseTensor<float>, int, DenseTensor<float>> environment,
        int                                         inferenceSteps)
    {
        int action  = Act(observation, goal, inferenceSteps);
        var nextObs = environment(observation, action);
        Observe(observation, action, nextObs, inferenceSteps);
        return (action, nextObs);
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    public float ComputeEnergy() => _worldModel.ComputeEnergy();

    public void FlushStore() => _worldModel.FlushStore();

    // ── Private helpers ───────────────────────────────────────────────────────

    /// <summary>Builds the world-model input vector [observation ⊕ one-hot(action)].</summary>
    private DenseTensor<float> BuildInput(DenseTensor<float> observation, int action)
    {
        if (observation.Length != _config.ObservationDim)
            throw new ArgumentException(
                $"observation has {observation.Length} elements; " +
                $"expected ObservationDim ({_config.ObservationDim}).");
        if (action < 0 || action >= _config.ActionCount)
            throw new ArgumentOutOfRangeException(nameof(action),
                $"action must be in [0, {_config.ActionCount - 1}].");

        var input = new DenseTensor<float>(_config.ObservationDim + _config.ActionCount);
        for (int i = 0; i < _config.ObservationDim; i++)
            input.Buffer[i] = observation.Buffer[i];
        input.Buffer[_config.ObservationDim + action] = 1f;
        return input;
    }

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose() => _worldModel.Dispose();
}
