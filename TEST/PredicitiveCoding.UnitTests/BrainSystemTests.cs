using PredictiveCoding.Brain;
using Xunit;

namespace PredictiveCoding.Brain.Tests;

/// <summary>Tests for Phase 7: ThousandBrainsSystem, SensorimotorAgent, and environment.</summary>
public class BrainSystemTests
{
    // ── PointCloudEnvironment ────────────────────────────────────────────────

    [Fact]
    public void PointCloudEnvironment_PresentAndObserve_ReturnsSameFeatureAsAdded()
    {
        var rng = new Random(1);
        var feat = Sdr.Random(40, 8, rng);
        var loc  = Sdr.Random(30, 6, rng);

        var env = new PointCloudEnvironment(seed: 1);
        env.AddObject(0, [(feat, loc)]);
        env.PresentObject(0);

        var (f, l) = env.Observe();

        // The returned SDRs must overlap completely with the originals
        // (same object — same bits in the SDR).
        Assert.Equal(feat.Overlap(feat), f.Overlap(feat));   // f IS feat
        Assert.Equal(loc.Overlap(loc),   l.Overlap(loc));
        Assert.Equal(0, env.CurrentObjectId);
    }

    [Fact]
    public void PointCloudEnvironment_MoveTo_CyclesThroughPoints()
    {
        var rng = new Random(2);
        var pts = Enumerable.Range(0, 4)
            .Select(_ => (Sdr.Random(40, 8, rng), Sdr.Random(30, 6, rng)))
            .ToList();

        var env = new PointCloudEnvironment(seed: 2);
        env.AddObject(0, pts);
        env.PresentObject(0);

        Assert.Equal(0, env.CurrentPointIndex);

        env.MoveTo(PointCloudEnvironment.MoveNext);
        Assert.Equal(1, env.CurrentPointIndex);

        env.MoveTo(PointCloudEnvironment.MovePrev);
        Assert.Equal(0, env.CurrentPointIndex);

        // Wrap-around backward from index 0
        env.MoveTo(PointCloudEnvironment.MovePrev);
        Assert.Equal(3, env.CurrentPointIndex);

        // Reset brings sensor back to start
        env.Reset();
        Assert.Equal(0, env.CurrentPointIndex);
    }

    // ── ThousandBrainsSystem ─────────────────────────────────────────────────

    [Fact]
    public void ThousandBrainsSystem_TeachObject_IncreasesLearnedObjectCount()
    {
        var sys = ThousandBrainsSystem.Create(MakeConfig());
        var rng = new Random(3);

        Assert.Equal(0, sys.LearnedObjectCount);

        int id0 = sys.TeachObject(MakeObs(4, rng));
        Assert.Equal(0, id0);
        Assert.Equal(1, sys.LearnedObjectCount);

        int id1 = sys.TeachObject(MakeObs(4, rng));
        Assert.Equal(1, id1);
        Assert.Equal(2, sys.LearnedObjectCount);
    }

    [Fact]
    public void ThousandBrainsSystem_ProcessObservation_ReturnsOutputForEveryArea()
    {
        var cfg = MakeConfig();
        var sys = ThousandBrainsSystem.Create(cfg);
        var rng = new Random(4);

        // Teach one object so there is something to infer against.
        sys.TeachObject(MakeObs(4, rng));

        var (feat, loc) = MakeObs(1, rng)[0];
        var outputs = sys.ProcessObservation(feat, loc);

        // One AreaOutput per sensory area.
        Assert.Equal(cfg.SensoryAreas, outputs.Count);
        Assert.All(outputs.Values, o =>
        {
            Assert.NotNull(o.Consensus);
            Assert.True(o.Round >= 0);
        });
    }

    // ── SensorimotorAgent ────────────────────────────────────────────────────

    [Fact]
    public void SensorimotorAgent_RunEpisode_StepsTakenWithinBudget()
    {
        var rng = new Random(5);
        var cfg = MakeConfig();
        var sys = ThousandBrainsSystem.Create(cfg);
        var env = new PointCloudEnvironment(seed: 5);

        const int nObjects = 3;
        const int pointsPerObject = 5;
        const int maxSteps = 8;

        // Register objects in the environment and teach the system.
        for (int o = 0; o < nObjects; o++)
        {
            var pts = MakeObs(pointsPerObject, rng);
            env.AddObject(o, pts);
            sys.TeachObject(pts);
        }

        // Run an episode on object 0.
        env.PresentObject(0);
        var agent  = new SensorimotorAgent(sys, new RandomPolicy(seed: 10));
        var result = agent.RunEpisode(env, maxSteps);

        Assert.True(result.StepsTaken >= 1, "Agent must take at least one step.");
        Assert.True(result.StepsTaken <= maxSteps, "Agent must not exceed step budget.");
        Assert.NotEmpty(result.HypothesisHistory);
        Assert.Equal(result.StepsTaken, result.HypothesisHistory.Count);
    }

    // ── UncertaintyPolicy ────────────────────────────────────────────────────

    [Fact]
    public void UncertaintyPolicy_ManyHypotheses_PrefersExploration()
    {
        const int availableActions = 3;
        var policy = new UncertaintyPolicy(
            explorationAction: PointCloudEnvironment.MoveRandom,
            exploitationAction: PointCloudEnvironment.MoveNext,
            uncertaintyScale: 2f,   // P(explore) = 1.0 at 3 hypotheses
            seed: 99);

        // With many hypotheses (uncertainty scale = 2 → P=1.0 at 3+),
        // every call should pick the exploration action.
        var manyHypotheses = new List<int> { 0, 1, 2, 3, 4 };
        int explorationCount = Enumerable.Range(0, 20)
            .Count(_ => policy.SelectAction(manyHypotheses, availableActions)
                        == PointCloudEnvironment.MoveRandom);

        // With 5 hypotheses and scale=2 → P(explore)=1 → all 20 should explore.
        Assert.Equal(20, explorationCount);

        // With one hypothesis (converged), policy must pick exploitation.
        var policy2 = new UncertaintyPolicy(seed: 7);
        int action = policy2.SelectAction([42], availableActions);
        Assert.Equal(PointCloudEnvironment.MoveNext % availableActions, action);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static ThousandBrainsConfig MakeConfig() => new()
    {
        SensoryAreas   = 2,
        ColumnsPerArea = 2,
        ColumnConfig   = new CorticalColumnConfig
        {
            InputSize      = 40,
            Columns        = 32,
            CellsPerColumn = 4,
            Sparsity       = 0.1f,
            MinOverlap     = 2,
            Seed           = 1,
        },
        MaxInferenceRounds = 4,
        Seed = 42,
    };

    private static List<(Sdr Feature, Sdr Location)> MakeObs(int count, Random rng)
        => Enumerable.Range(0, count)
            .Select(_ => (Sdr.Random(40, 8, rng), Sdr.Random(30, 6, rng)))
            .ToList();
}
