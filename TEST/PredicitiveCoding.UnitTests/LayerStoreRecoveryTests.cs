using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// End-to-end persistence/recovery tests for <see cref="LayerStore"/> through the
/// public <see cref="Network"/> API. They prove that reads come straight from
/// FasterKV (there is no shadow byte[] cache) and that a store reopened on the
/// same path recovers the trained weights instead of reinitializing.
/// </summary>
public sealed class LayerStoreRecoveryTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"pc_recover_{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_tmp)) Directory.Delete(_tmp, true);
    }

    private static NetworkConfig ConfigAt(string path) => new()
    {
        Path = path,
        LayerSizes = [8, 16, 4],
        Activation = ActivationType.Tanh,
        StateLearningRate = 0.1f,
        WeightLearningRate = 0.05f,
    };

    private static (DenseTensor<float> x, DenseTensor<float> y) Sample()
    {
        var x = new DenseTensor<float>(8);
        for (int i = 0; i < 8; i++) x[i] = MathF.Sin(i + 1);
        var y = new DenseTensor<float>(4);
        y[2] = 1f;
        return (x, y);
    }

    [Fact]
    public void Reopen_RecoversTrainedWeights_NotReinitialized()
    {
        var (x, y) = Sample();

        // Train network A (seed 1) and record its trained weight norm.
        float wnTrained;
        using (var a = new Network(ConfigAt(_tmp), new Random(1)))
        {
            for (int it = 0; it < 80; it++) a.Learn(x, y, inferenceSteps: 8);
            a.FlushStore();
            wnTrained = a.AverageWeightNorm();
        } // Dispose → full checkpoint on disk

        // Reopen on the SAME path but with a DIFFERENT seed: if recovery works,
        // the weights come from disk and the seed is irrelevant.
        float wnRecovered;
        using (var b = new Network(ConfigAt(_tmp), new Random(999)))
            wnRecovered = b.AverageWeightNorm();

        // Sensitivity guard: a fresh store on a different path with that same
        // seed yields a clearly different norm — so the equality below is real.
        float wnFresh;
        string other = _tmp + "_fresh";
        try
        {
            using var c = new Network(ConfigAt(other), new Random(999));
            wnFresh = c.AverageWeightNorm();
        }
        finally
        {
            if (Directory.Exists(other)) Directory.Delete(other, true);
        }

        Assert.Equal(wnTrained, wnRecovered, 4);   // exact weights recovered from FasterKV
        Assert.True(MathF.Abs(wnTrained - wnFresh) > 1e-3f,
            $"signal not sensitive: trained={wnTrained}, fresh={wnFresh}");
    }

    [Fact]
    public void Reopen_PreservesPredictions()
    {
        var (x, y) = Sample();

        float energyBefore;
        using (var a = new Network(ConfigAt(_tmp), new Random(7)))
        {
            for (int it = 0; it < 80; it++) a.Learn(x, y, inferenceSteps: 8);
            a.FlushStore();
            (_, energyBefore) = a.PredictWithEnergy(x, inferenceSteps: 8);
        }

        float energyAfter;
        using (var b = new Network(ConfigAt(_tmp), new Random(123)))
            (_, energyAfter) = b.PredictWithEnergy(x, inferenceSteps: 8);

        // Identical weights ⇒ identical inference energy for the same input.
        Assert.Equal(energyBefore, energyAfter, 3);
    }
}
