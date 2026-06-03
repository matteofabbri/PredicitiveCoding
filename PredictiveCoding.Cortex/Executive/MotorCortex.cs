using PredictiveCoding.Core;
using PredictiveCoding.Expanded.Active;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// MotorCortex  —  active-inference motor control (perception–action loop)
//
// In active inference motor commands are selected to make preferred (goal)
// observations come true: the cortex runs a generative world model and picks the
// action minimising expected free energy toward the goal.  MotorCortex wraps the
// Active-PC agent and adds a closed-loop controller that drives an external plant
// toward a goal observation.
//
// References: Adams, Shipp & Friston (2013) PC of action; Friston et al. (2017).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record MotorCortexConfig
{
    /// <summary>Active-inference agent (world model + policy).</summary>
    public ActiveNetworkConfig Agent { get; init; } = new();

    /// <summary>L2 distance below which the goal is considered reached.</summary>
    public float GoalTolerance { get; init; } = 0.1f;

    public string BasePath { get; init; } = "./cortex/motor";

    public void Validate()
    {
        Agent.Validate();
        if (GoalTolerance < 0f)
            throw new InvalidOperationException("GoalTolerance must be >= 0.");
    }
}

public sealed class MotorCortex : IDisposable
{
    private readonly ActiveNetwork    _agent;
    private readonly MotorCortexConfig _config;

    public int ObservationDim => _agent.ObservationDim;
    public int ActionCount    => _agent.ActionCount;

    public MotorCortex(MotorCortexConfig config, Random rng)
    {
        config.Validate();
        _config = config;
        _agent  = new ActiveNetwork(config.Agent with { BasePath = config.BasePath }, rng);
    }

    /// <summary>Predicts the proprioceptive consequence of a motor command.</summary>
    public DenseTensor<float> PredictOutcome(DenseTensor<float> state, int action, int steps)
        => _agent.PredictNext(state, action, steps);

    /// <summary>Selects the motor command minimising expected free energy toward the goal.</summary>
    public int SelectCommand(DenseTensor<float> state, DenseTensor<float> goal, int steps)
        => _agent.Act(state, goal, steps);

    /// <summary>Learns one transition (state, command) → next state.</summary>
    public void Learn(DenseTensor<float> state, int action, DenseTensor<float> nextState, int steps)
        => _agent.Observe(state, action, nextState, steps);

    /// <summary>
    /// Closed-loop reach: repeatedly select a command toward <paramref name="goal"/>,
    /// apply it to <paramref name="plant"/>, and learn from the outcome, until the
    /// goal is reached or <paramref name="maxSteps"/> is hit.  Returns the visited
    /// states (including the start).
    /// </summary>
    public List<DenseTensor<float>> ReachGoal(
        DenseTensor<float>                                start,
        DenseTensor<float>                                goal,
        Func<DenseTensor<float>, int, DenseTensor<float>> plant,
        int                                               maxSteps,
        int                                               inferenceSteps)
    {
        var trajectory = new List<DenseTensor<float>> { start.Clone() };
        var state      = start;

        for (int t = 0; t < maxSteps; t++)
        {
            if (Distance(state, goal) <= _config.GoalTolerance) break;
            var (action, next) = _agent.Step(state, goal, plant, inferenceSteps);
            _ = action;
            trajectory.Add(next.Clone());
            state = next;
        }
        return trajectory;
    }

    public void FlushAll() => _agent.FlushStore();
    public void Dispose()  => _agent.Dispose();

    private static float Distance(DenseTensor<float> a, DenseTensor<float> b)
        => MathF.Sqrt(TensorUtils.SumSquares(TensorUtils.ElementwiseSub(a, b)) * 2f);
}
