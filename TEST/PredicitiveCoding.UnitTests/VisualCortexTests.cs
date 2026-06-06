using PredictiveCoding.Core;
using PredictiveCoding.Cortex;
using PredictiveCoding.Expanded.Hierarchical;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

public class VisualCortexTests : IDisposable
{
    private readonly string _tempDir;

    public VisualCortexTests()
        => _tempDir = Path.Combine(Path.GetTempPath(), $"pccortex_{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true);
    }

    private VisualCortex Make() => new(
        new VisualCortexConfig
        {
            ImageChannels = 1, ImageHeight = 4, ImageWidth = 4,
            PatchSize = 2, PatchStride = 2,                       // 2×2 grid
            V1Column    = new NetworkConfig { LayerSizes = [4, 8, 4] },   // K = 4
            HigherAreas = new HierarchicalNetworkConfig
            {
                Modules =
                [
                    new NetworkConfig { LayerSizes = [2 * 2 * 4, 8] },    // 16 → 8
                    new NetworkConfig { LayerSizes = [8, 4] },           // → IT = 4
                ],
            },
            Classifier = new NetworkConfig { LayerSizes = [4, 2] },
            BasePath   = _tempDir,
        },
        new Random(42));

    private static DenseTensor<float> Image()
    {
        var t = new DenseTensor<float>(1, 4, 4);
        for (int i = 0; i < t.Length; i++) t.Buffer[i] = (i % 3 - 1) * 0.5f;
        return t;
    }

    [Fact]
    public void Encode_ReturnsItLatent()
    {
        using var cortex = Make();
        Assert.Equal(4, cortex.ItLatentDim);
        Assert.Equal(4, cortex.Encode(Image(), 8).Length);
    }

    [Fact]
    public void EncodeAreas_ReturnsV1PlusHigherAreas()
    {
        using var cortex = Make();
        var areas = cortex.EncodeAreas(Image(), 8);
        Assert.Equal(3, areas.Length);          // V1 map + 2 higher areas
        Assert.Equal(2 * 2 * 4, areas[0].Length); // V1 feature map
        Assert.Equal(4, areas[^1].Length);        // IT
    }

    [Fact]
    public void Reconstruct_HasImageShape()
    {
        using var cortex = Make();
        Assert.Equal(1 * 4 * 4, cortex.Reconstruct(Image(), 8).Length);
    }

    [Fact]
    public void Predict_ReturnsClassInRange()
    {
        using var cortex = Make();
        var img = Image();
        cortex.LearnSupervised(img, new DenseTensor<float>([1f, 0f], 2), 8);
        Assert.InRange(cortex.Predict(img, 8), 0, 1);
    }
}
