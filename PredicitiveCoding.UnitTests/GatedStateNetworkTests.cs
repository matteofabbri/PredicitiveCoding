using PredictiveCoding.Core;
using PredictiveCoding.Expanded.Gated;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

public class GatedStateNetworkTests : IDisposable
{
    private readonly string _tempDir;

    public GatedStateNetworkTests()
        => _tempDir = Path.Combine(Path.GetTempPath(), $"pcgated_{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true);
    }

    private GatedStateNetwork Make() => new(
        new GatedStateNetworkConfig
        {
            NetworkConfig = new NetworkConfig { LayerSizes = [4, 8, 3], Path = _tempDir },
            BasePath      = _tempDir,
        },
        new Random(42));

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    [Fact]
    public void MemoryDim_MatchesTopLayer()
    {
        using var net = Make();
        Assert.Equal(3, net.MemoryDim);
    }

    [Fact]
    public void Step_PopulatesMemory_AndResetClears()
    {
        using var net = Make();
        Assert.False(net.HasMemory);

        var m = net.Step(Vec(0.5f, -0.2f, 0.1f, 0.3f), 8);
        Assert.True(net.HasMemory);
        Assert.Equal(3, m.Length);
        Assert.NotNull(net.MemoryState);

        net.Reset();
        Assert.False(net.HasMemory);
        Assert.Null(net.MemoryState);
    }

    [Fact]
    public void Predict_ReturnsClassInRange()
    {
        using var net = Make();
        var x = Vec(0.5f, -0.2f, 0.1f, 0.3f);
        net.Step(x, 8);
        Assert.InRange(net.Predict(x, 8), 0, 2);
    }
}
