using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

// Phase 1 of PredictiveCoding.LLM/ROADMAP.md: adaptive inference effort.
// "Effort" is the number of settling steps; the free-energy trajectory is the
// native early-exit signal.
public class AdaptiveInferenceTests : IDisposable
{
    private readonly string        _tempDir;
    private readonly NetworkConfig _cfg;

    public AdaptiveInferenceTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"pcadapt_{Guid.NewGuid():N}");
        _cfg = new NetworkConfig
        {
            Path               = _tempDir,
            LayerSizes         = [4, 8, 2],
            Activation         = ActivationType.Tanh,
            StateLearningRate  = 0.1f,
            WeightLearningRate = 0.01f,
        };
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }

    private Network Net() => new(_cfg, new Random(42));

    private static DenseTensor<float> Input(params float[] values) => new(values, values.Length);

    private static DenseTensor<float> OneHot(int index, int size)
    {
        var t = DenseTensor<float>.Zeros(size);
        t.Buffer[index] = 1f;
        return t;
    }

    /// <summary>Two clearly separable patterns, learned until they are easy.</summary>
    private static void TrainTwoPatterns(Network net)
    {
        var a = Input(1f, 0f, 0f, 0f);
        var b = Input(0f, 0f, 0f, 1f);
        for (int epoch = 0; epoch < 40; epoch++)
        {
            net.Learn(a, OneHot(0, 2), 30);
            net.Learn(b, OneHot(1, 2), 30);
        }
    }

    // ── ε = 0 ⇒ the fixed-step path, bit for bit ─────────────────────────────

    [Fact]
    public void InferAdaptive_EpsilonZero_MatchesInferGetTop_BitForBit()
    {
        using var net = Net();
        TrainTwoPatterns(net);
        var x = Input(1f, 0f, 0f, 0f);

        var reference             = net.InferGetTop(x, 20, resetState: true);
        var (top, steps, energy)  = net.InferAdaptive(x, 20, epsilon: 0f);

        Assert.Equal(20, steps);                       // early exit disabled
        Assert.True(float.IsFinite(energy) && energy >= 0f);
        for (int i = 0; i < reference.Buffer.Length; i++)
            Assert.Equal(reference.Buffer[i], top.Buffer[i]);   // exact float equality
    }

    [Fact]
    public void PredictAdaptive_EpsilonZero_MatchesPredictWithEnergy()
    {
        using var net = Net();
        TrainTwoPatterns(net);
        var x = Input(0f, 0f, 0f, 1f);

        var (cls, energy)                 = net.PredictWithEnergy(x, 15);
        var (aCls, aEnergy, aStepsUsed)   = net.PredictAdaptive(x, 15, epsilon: 0f);

        Assert.Equal(cls, aCls);
        Assert.Equal(energy, aEnergy, precision: 5);
        Assert.Equal(15 * (2 + 1), aStepsUsed);        // every candidate + winner re-run at full budget
    }

    // ── Early exit: fires when the descent flattens, and only then ───────────
    //
    // Note what is certified here: properties guaranteed by the exit rule
    // itself. How many steps a given input spends is a fact about the energy
    // landscape, not the rule — empirically a well-trained input can settle
    // LONGER than noise, because its descent stays productive while noise hits
    // a shallow "nothing more to explain" minimum almost immediately.

    [Fact]
    public void InferAdaptive_EarlyExit_FiresAndIsEpsilonMonotonicAndBudgetInvariant()
    {
        using var net = Net();
        TrainTwoPatterns(net);
        var x = Input(1f, 0f, 0f, 0f);

        var (_, loose, _) = net.InferAdaptive(x, 60, epsilon: 1e-2f);
        var (_, tight, _) = net.InferAdaptive(x, 60, epsilon: 1e-4f);
        var (_, full,  _) = net.InferAdaptive(x, 60, epsilon: 0f);

        Assert.True(loose < 60, $"expected an early exit, spent the whole budget ({loose})");
        Assert.True(loose <= tight, $"looser tolerance must stop no later ({loose} vs {tight})");
        Assert.Equal(60, full);                        // ε = 0 ⇒ whole budget

        var (_, samePoint, _) = net.InferAdaptive(x, 200, epsilon: 1e-2f);
        Assert.Equal(loose, samePoint);                // exit step depends on the trajectory, not the budget
    }

    [Fact]
    public void PredictAdaptive_WithEpsilon_SpendsFewerStepsAndStaysCorrect()
    {
        using var net = Net();
        TrainTwoPatterns(net);

        // Budget generously above the convergence point, so the early exit is
        // guaranteed to fire and the savings become visible.
        var (cls, energy, stepsUsed) = net.PredictAdaptive(Input(1f, 0f, 0f, 0f), 200, epsilon: 1e-3f);

        Assert.Equal(0, cls);
        Assert.True(float.IsFinite(energy) && energy >= 0f);
        Assert.True(stepsUsed < 200 * (2 + 1), $"expected adaptive savings, spent {stepsUsed}/600");
    }

    // ── Effort presets ───────────────────────────────────────────────────────

    [Fact]
    public void EffortPresets_StepsGrowAndToleranceTightens()
    {
        var low    = EffortPresets.For(EffortLevel.Low);
        var medium = EffortPresets.For(EffortLevel.Medium);
        var high   = EffortPresets.For(EffortLevel.High);

        Assert.True(low.MaxSteps < medium.MaxSteps && medium.MaxSteps < high.MaxSteps);
        Assert.True(low.Epsilon  > medium.Epsilon  && medium.Epsilon  > high.Epsilon);
    }

    [Fact]
    public void InferAdaptive_EffortOverload_RespectsTheBudget()
    {
        using var net = Net();
        TrainTwoPatterns(net);

        var (top, steps, _) = net.InferAdaptive(Input(1f, 0f, 0f, 0f), EffortLevel.Low);

        Assert.Equal(_cfg.LayerSizes[^1], top.Length);
        Assert.InRange(steps, 1, EffortPresets.For(EffortLevel.Low).MaxSteps);
    }

    // ── Argument validation ──────────────────────────────────────────────────

    [Fact]
    public void InferAdaptive_NonPositiveMaxSteps_Throws()
    {
        using var net = Net();
        Assert.Throws<ArgumentOutOfRangeException>(() => net.InferAdaptive(Input(1f, 0f, 0f, 0f), 0, 1e-3f));
        Assert.Throws<ArgumentOutOfRangeException>(() => net.PredictAdaptive(Input(1f, 0f, 0f, 0f), 0, 1e-3f));
    }
}
