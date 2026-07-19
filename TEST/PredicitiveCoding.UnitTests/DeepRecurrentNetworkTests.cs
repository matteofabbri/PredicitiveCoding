using PredictiveCoding.Core;
using PredictiveCoding.Networks.DeepRecurrent;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

public class DeepRecurrentNetworkTests : IDisposable
{
    private readonly string _tempDir;

    public DeepRecurrentNetworkTests()
        => _tempDir = Path.Combine(Path.GetTempPath(), $"pcdrn_{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true);
    }

    private DeepRecurrentNetwork Make() => new(
        new DeepRecurrentNetworkConfig
        {
            BasePath = _tempDir,
            Stages =
            [
                new NetworkConfig { LayerSizes = [4, 6, 3] },
                new NetworkConfig { LayerSizes = [3, 5, 2] },
            ],
            TemporalPrecision = [1f],
        },
        new Random(42));

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    [Fact]
    public void Dimensions_AreExposedCorrectly()
    {
        using var net = Make();
        Assert.Equal(2, net.StageCount);
        Assert.Equal(4, net.InputDim);
        Assert.Equal(2, net.TopLatentDim);
    }

    [Fact]
    public void Step_ReturnsTopLatent_AndBuildsContext()
    {
        using var net = Make();
        Assert.False(net.HasContext);

        var top = net.Step(Vec(0.2f, -0.1f, 0.4f, 0.0f), 8);
        Assert.Equal(2, top.Length);
        Assert.True(net.HasContext);
    }

    [Fact]
    public void Reset_ClearsContext()
    {
        using var net = Make();
        net.Step(Vec(0.2f, -0.1f, 0.4f, 0.0f), 8);
        net.Reset();
        Assert.False(net.HasContext);
    }

    [Fact]
    public void Encode_RollsContextWithoutThrowing()
    {
        using var net = Make();
        net.Step(Vec(0.2f, -0.1f, 0.4f, 0.0f), 8);
        var top = net.Encode(Vec(0.1f, 0.1f, 0.1f, 0.1f), 8);
        Assert.Equal(2, top.Length);
    }
}
