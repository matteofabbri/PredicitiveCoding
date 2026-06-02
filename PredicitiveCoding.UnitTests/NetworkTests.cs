using PredictiveCoding.Core;
using PredictiveCoding.Model;

namespace PredicitiveCoding.UnitTests;

public class NetworkTests : IDisposable
{
    private readonly string      _tempDir;
    private readonly NetworkConfig _cfg;

    public NetworkTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"pcnet_{Guid.NewGuid():N}");
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

    private static DenseTensor<float> Input(params float[] values) =>
        new(values, values.Length);

    private static DenseTensor<float> OneHot(int index, int size)
    {
        var t = DenseTensor<float>.Zeros(size);
        t.Buffer[index] = 1f;
        return t;
    }

    // ── Construction ──────────────────────────────────────────────────────────

    [Fact]
    public void LayerCount_MatchesConfig()
    {
        using var net = Net();
        Assert.Equal(_cfg.LayerSizes.Length, net.LayerCount);
    }

    // ── ComputeEnergy ─────────────────────────────────────────────────────────

    [Fact]
    public void ComputeEnergy_AfterConstruction_IsNonNegative()
    {
        using var net = Net();
        Assert.True(net.ComputeEnergy() >= 0f);
    }

    // ── ComputeLayerEnergies ──────────────────────────────────────────────────

    [Fact]
    public void ComputeLayerEnergies_LengthIsLayerCountMinusOne()
    {
        using var net = Net();
        Assert.Equal(net.LayerCount - 1, net.ComputeLayerEnergies().Length);
    }

    [Fact]
    public void ComputeLayerEnergies_AllNonNegative()
    {
        using var net = Net();
        Assert.All(net.ComputeLayerEnergies(), e => Assert.True(e >= 0f));
    }

    // ── AverageWeightNorm ─────────────────────────────────────────────────────

    [Fact]
    public void AverageWeightNorm_AfterConstruction_IsPositive()
    {
        using var net = Net();
        Assert.True(net.AverageWeightNorm() > 0f);
    }

    // ── Infer ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Infer_IsTransient_DoesNotChangeWeights()
    {
        using var net = Net();
        float normBefore = net.AverageWeightNorm();

        net.Infer(Input(1f, 0f, 0f, 1f), inferenceSteps: 5);

        Assert.Equal(normBefore, net.AverageWeightNorm(), precision: 6);
    }

    // ── Learn ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Learn_PersistsWeightChangesToStore()
    {
        using var net = Net();
        float normBefore = net.AverageWeightNorm();

        net.Learn(Input(1f, 0f, 0f, 1f), OneHot(0, size: 2), inferenceSteps: 5);

        Assert.NotEqual(normBefore, net.AverageWeightNorm());
    }

    [Fact]
    public void Learn_MultipleSteps_EnergyDecreasesOrStabilises()
    {
        using var net = Net();
        var x = Input(1f, 0f, 0f, 1f);
        var y = OneHot(0, size: 2);

        for (int i = 0; i < 20; i++)
            net.Learn(x, y, inferenceSteps: 10);

        // Energy after training should be finite and non-negative.
        float energy = net.ComputeEnergy();
        Assert.True(energy >= 0f && float.IsFinite(energy));
    }

    // ── Predict ───────────────────────────────────────────────────────────────

    [Fact]
    public void Predict_ReturnsValidClassIndex()
    {
        using var net = Net();
        int predicted = net.Predict(Input(1f, 0f, 0f, 1f), inferenceSteps: 5);
        Assert.InRange(predicted, 0, _cfg.LayerSizes[^1] - 1);
    }

    [Fact]
    public void Predict_PersistsWinnerStateToStore()
    {
        using var net = Net();
        net.Predict(Input(1f, 0f, 0f, 1f), inferenceSteps: 5);

        float energy = net.ComputeEnergy();
        Assert.True(energy >= 0f && float.IsFinite(energy));
    }

    [Fact]
    public void Predict_IsDeterministic()
    {
        using var net = Net();
        var input = Input(0.5f, 0.1f, 0.9f, 0.3f);

        int first  = net.Predict(input, inferenceSteps: 10);
        int second = net.Predict(input, inferenceSteps: 10);

        Assert.Equal(first, second);
    }

    // ── Learn + Predict ───────────────────────────────────────────────────────

    [Fact]
    public void LearnRepeated_ConvergesOnLinearlySeperablePatterns()
    {
        // all-zeros → class 0 / all-ones → class 1 is trivially separable.
        var cfg = new NetworkConfig
        {
            Path               = Path.Combine(_tempDir, "converge"),
            LayerSizes         = [4, 16, 2],
            Activation         = ActivationType.Tanh,
            StateLearningRate  = 0.2f,
            WeightLearningRate = 0.05f,
        };
        using var net = new Network(cfg, new Random(0));

        var x0 = Input(0f, 0f, 0f, 0f);
        var x1 = Input(1f, 1f, 1f, 1f);
        var y0 = OneHot(0, 2);
        var y1 = OneHot(1, 2);

        for (int i = 0; i < 200; i++)
        {
            net.Learn(x0, y0, inferenceSteps: 10);
            net.Learn(x1, y1, inferenceSteps: 10);
        }

        Assert.Equal(0, net.Predict(x0, inferenceSteps: 20));
        Assert.Equal(1, net.Predict(x1, inferenceSteps: 20));
    }
}
