using PredictiveCoding.Core;
using PredictiveCoding.Expanded.Hierarchical;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

public class HierarchicalNetworkTests : IDisposable
{
    private readonly string _tempDir;

    public HierarchicalNetworkTests()
        => _tempDir = Path.Combine(Path.GetTempPath(), $"pchier_{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true);
    }

    private HierarchicalNetwork Make(ActivationType act = ActivationType.Tanh) => new(
        new HierarchicalNetworkConfig
        {
            BasePath = _tempDir,
            Modules =
            [
                new NetworkConfig { LayerSizes = [6, 8, 4], Activation = act, WeightLearningRate = 0.05f },
                new NetworkConfig { LayerSizes = [4, 3],    Activation = act, WeightLearningRate = 0.05f },
            ],
        },
        new Random(42));

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    [Fact]
    public void Dimensions_AreExposedCorrectly()
    {
        using var net = Make();
        Assert.Equal(2, net.ModuleCount);
        Assert.Equal(6, net.InputDim);
        Assert.Equal(3, net.TopLatentDim);
    }

    [Fact]
    public void Encode_ReturnsTopLatent_AndEncodeAllReturnsPerModule()
    {
        using var net = Make();
        var x = Vec(1, 0, -1, 0.5f, 0.2f, -0.3f);

        Assert.Equal(3, net.Encode(x, 10).Length);

        var all = net.EncodeAll(x, 10);
        Assert.Equal(2, all.Length);
        Assert.Equal(4, all[0].Length);
        Assert.Equal(3, all[1].Length);
    }

    [Fact]
    public void Reconstruct_HasInputShape()
    {
        using var net = Make();
        var x = Vec(1, 0, -1, 0.5f, 0.2f, -0.3f);
        Assert.Equal(6, net.Reconstruct(x, 10).Length);
    }

    [Fact]
    public void Learning_ReducesReconstructionError()
    {
        using var net = Make(ActivationType.Linear);
        var x = Vec(0.6f, -0.4f, 0.2f, 0.1f, -0.5f, 0.3f);

        float before = net.ReconstructionError(x, 15);
        for (int i = 0; i < 60; i++) net.Learn(x, 15);
        float after = net.ReconstructionError(x, 15);

        Assert.True(after < before, $"expected error to drop (before={before}, after={after}).");
    }
}
