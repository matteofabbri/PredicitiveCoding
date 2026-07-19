using PredictiveCoding.Core;
using PredictiveCoding.Cortex;
using PredictiveCoding.Networks.Active;
using PredictiveCoding.Networks;

namespace PredictiveCoding.CortexApp.Examples;

// A 2-D navigation agent (MotorCortex / Active PC): learns a world model by random
// exploration, then reaches a goal by minimising expected free energy.
public static class ActiveInferenceExample
{
    private const float Delta = 0.12f;   // step size per action

    public sealed record Result(
        IReadOnlyList<(float X, float Y)> Trajectory,
        bool                              Reached,
        float                             FinalDistance,
        int                               ExploreSteps);

    public static Result Run(int exploreSteps, int inferenceSteps)
    {
        var rng = new Random(5);
        var cfg = new MotorCortexConfig
        {
            Agent = new ActiveNetworkConfig
            {
                ObservationDim   = 2,
                ActionCount      = 4,
                WorldModelConfig = new NetworkConfig
                {
                    LayerSizes         = [6, 16, 2],          // [x,y ⊕ one-hot(4)] → [x,y]
                    Activation         = ActivationType.Tanh,
                    WeightLearningRate = 0.05f,
                },
            },
            GoalTolerance = 0.1f,
            BasePath      = Path.Combine(Path.GetTempPath(), "cortexapp", Guid.NewGuid().ToString("N")),
        };

        using var motor = new MotorCortex(cfg, rng);

        // Plant: apply an action to a position, clamped to the unit square.
        DenseTensor<float> Plant(DenseTensor<float> s, int a)
        {
            float x = s.Buffer[0], y = s.Buffer[1];
            switch (a)
            {
                case 0: x += Delta; break;
                case 1: x -= Delta; break;
                case 2: y += Delta; break;
                default: y -= Delta; break;
            }
            return Pos(Math.Clamp(x, 0f, 1f), Math.Clamp(y, 0f, 1f));
        }

        // Learn a world model from random exploration.
        var state = Pos(0.5f, 0.5f);
        for (int i = 0; i < exploreSteps; i++)
        {
            int a    = rng.Next(4);
            var next = Plant(state, a);
            motor.Learn(state, a, next, inferenceSteps);
            state = rng.NextDouble() < 0.1 ? Pos((float)rng.NextDouble(), (float)rng.NextDouble()) : next;
        }

        // Goal-directed reach.
        var start = Pos(0.1f, 0.1f);
        var goal  = Pos(0.9f, 0.9f);
        var path  = motor.ReachGoal(start, goal, Plant, maxSteps: 40, inferenceSteps);

        var traj = path.Select(p => (p.Buffer[0], p.Buffer[1])).ToList();
        var last = path[^1];
        float dist = MathF.Sqrt((last.Buffer[0] - goal.Buffer[0]) * (last.Buffer[0] - goal.Buffer[0]) +
                                (last.Buffer[1] - goal.Buffer[1]) * (last.Buffer[1] - goal.Buffer[1]));

        try { Directory.Delete(cfg.BasePath, recursive: true); } catch { /* ignore */ }
        return new Result(traj, dist <= cfg.GoalTolerance, dist, exploreSteps);
    }

    private static DenseTensor<float> Pos(float x, float y) => new([x, y], 2);
}
