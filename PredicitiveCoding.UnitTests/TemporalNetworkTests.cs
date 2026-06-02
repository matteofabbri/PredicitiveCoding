using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

public sealed class TemporalNetworkTests : IDisposable
{
    private readonly string _tempDir;

    public TemporalNetworkTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"tmpnet_{Guid.NewGuid():N}");
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }

    private TemporalNetwork Build(string sub = "default", float precision = 1f)
    {
        var cfg = new NetworkConfig
        {
            Path               = Path.Combine(_tempDir, sub),
            LayerSizes         = [4, 8, 4],
            Activation         = ActivationType.Tanh,
            StateLearningRate  = 0.1f,
            WeightLearningRate = 0.01f,
        };
        return new TemporalNetwork(new Network(cfg, new Random(42)), precision);
    }

    private static DenseTensor<float> OneHot(int cls, int size = 4)
    {
        var t = DenseTensor<float>.Zeros(size);
        t.Buffer[cls] = 1f;
        return t;
    }

    // ── Construction ─────────────────────────────────────────────────────────

    [Fact]
    public void Construction_ValidArgs_DoesNotThrow()
    {
        using var tnet = Build();
        Assert.False(tnet.HasPrior);
    }

    [Fact]
    public void Construction_NegativePrecision_Throws()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => Build(precision: -0.1f));
    }

    [Fact]
    public void Construction_TwoLayerNetwork_Throws()
    {
        var cfg = new NetworkConfig
        {
            Path       = Path.Combine(_tempDir, "2layer"),
            LayerSizes = [4, 4],
        };
        Assert.Throws<ArgumentException>(
            () => new TemporalNetwork(new Network(cfg, new Random(0))));
    }

    // ── Reset ────────────────────────────────────────────────────────────────

    [Fact]
    public void Reset_ClearsPrior()
    {
        using var tnet = Build();
        tnet.Step(OneHot(0), OneHot(0), inferenceSteps: 3);
        Assert.True(tnet.HasPrior);

        tnet.Reset();
        Assert.False(tnet.HasPrior);
    }

    // ── Step ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Step_FirstCall_NoPrior_DoesNotThrow()
    {
        using var tnet = Build();
        tnet.Step(OneHot(0), OneHot(0), inferenceSteps: 5);
        Assert.True(tnet.HasPrior);
    }

    [Fact]
    public void Step_MultipleCalls_EnergyFiniteAndNonNegative()
    {
        using var tnet = Build();
        for (int t = 0; t < 8; t++)
        {
            int cls = t % 4;
            tnet.Step(OneHot(cls), OneHot(cls), inferenceSteps: 5);
            float e = tnet.BaseNetwork.ComputeEnergy();
            Assert.True(e >= 0f && float.IsFinite(e));
        }
    }

    [Fact]
    public void Step_WeightsChangeAfterTraining()
    {
        using var tnet = Build();
        float normBefore = tnet.BaseNetwork.AverageWeightNorm();

        tnet.Step(OneHot(0), OneHot(0), inferenceSteps: 5);

        Assert.NotEqual(normBefore, tnet.BaseNetwork.AverageWeightNorm());
    }

    // ── Predict ───────────────────────────────────────────────────────────────

    [Fact]
    public void Predict_WithoutPrior_ReturnsValidClass()
    {
        using var tnet = Build();
        int pred = tnet.Predict(OneHot(0), inferenceSteps: 5);
        Assert.InRange(pred, 0, 3);
    }

    [Fact]
    public void Predict_WithPrior_ReturnsValidClass()
    {
        using var tnet = Build();
        tnet.Step(OneHot(0), OneHot(0), inferenceSteps: 5);
        int pred = tnet.Predict(OneHot(1), inferenceSteps: 5);
        Assert.InRange(pred, 0, 3);
    }

    // ── Advance ───────────────────────────────────────────────────────────────

    [Fact]
    public void Advance_DoesNotChangeWeights()
    {
        using var tnet = Build();
        for (int i = 0; i < 20; i++)
            tnet.Step(OneHot(i % 4), OneHot(i % 4), inferenceSteps: 5);

        float normBefore = tnet.BaseNetwork.AverageWeightNorm();

        tnet.Advance(OneHot(0), OneHot(0), inferenceSteps: 5);

        Assert.Equal(normBefore, tnet.BaseNetwork.AverageWeightNorm(), precision: 5);
    }

    [Fact]
    public void Advance_AdvancesPrior()
    {
        using var tnet = Build();
        tnet.Step(OneHot(0), OneHot(0), inferenceSteps: 5);
        Assert.True(tnet.HasPrior);

        // After a reset and advance from cold, prior should still be populated.
        tnet.Reset();
        tnet.Advance(OneHot(1), OneHot(1), inferenceSteps: 5);
        Assert.True(tnet.HasPrior);
    }

    // ── Temporal precision = 0 degenerates to standard network ───────────────

    [Fact]
    public void ZeroPrecision_MatchesStandardPredict()
    {
        string pathTemporal = Path.Combine(_tempDir, "zero_t");
        string pathBase     = Path.Combine(_tempDir, "zero_b");

        var cfg = new NetworkConfig
        {
            LayerSizes         = [4, 8, 4],
            Activation         = ActivationType.Tanh,
            StateLearningRate  = 0.1f,
            WeightLearningRate = 0.01f,
        };

        // Train identically
        using var baseNet  = new Network(cfg with { Path = pathBase },    new Random(7));
        using var temporal = new TemporalNetwork(
            new Network(cfg with { Path = pathTemporal }, new Random(7)), temporalPrecision: 0f);

        for (int i = 0; i < 50; i++)
        {
            int cls   = i % 4;
            var input = OneHot(cls);
            var label = OneHot(cls);
            baseNet.Learn(input, label, inferenceSteps: 10);
            temporal.Step(input, label, inferenceSteps: 10);
        }

        // With Π_t=0 the temporal term vanishes, so predictions should agree.
        for (int cls = 0; cls < 4; cls++)
        {
            int basePred = baseNet.Predict(OneHot(cls), inferenceSteps: 20);
            int tempPred = temporal.Predict(OneHot(cls), inferenceSteps: 20);
            Assert.Equal(basePred, tempPred);
        }
    }

    // ── Sequence learning convergence ────────────────────────────────────────

    [Fact]
    public void TemporalNetwork_LearnsCyclicSequence()
    {
        using var tnet = Build("cycle", precision: 2f);

        // Train: 0→1→2→3→0→...  clean (no noise) for 300 cycles
        for (int epoch = 0; epoch < 300; epoch++)
        {
            tnet.Reset();
            for (int t = 0; t < 4; t++)
                tnet.Step(OneHot(t), OneHot(t), inferenceSteps: 15);
        }

        // Evaluate: given clean input at each position, correct class expected
        int correct = 0;
        for (int rep = 0; rep < 10; rep++)
        {
            tnet.Reset();
            for (int t = 0; t < 4; t++)
            {
                int pred = tnet.Advance(OneHot(t), OneHot(t), inferenceSteps: 15);
                if (pred == t) correct++;
            }
        }

        // Expect at least 75% accuracy (3/4 positions correct on average)
        Assert.True(correct >= 30, $"Expected ≥30/40 correct, got {correct}/40");
    }
}
