using PredictiveCoding.Core;
using PredictiveCoding.Cortex;
using PredictiveCoding.Expanded.Hierarchical;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

// Group E — extra memory: PerirhinalCortex, ParahippocampalCortex.
public class CortexMemoryExtraTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"memextra_{Guid.NewGuid():N}");

    public void Dispose() { if (Directory.Exists(_tmp)) Directory.Delete(_tmp, recursive: true); }

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    private static DenseTensor<float> SceneImg(float scale)
    {
        var t = new DenseTensor<float>(1, 4, 4);
        for (int i = 0; i < t.Length; i++) t.Buffer[i] = (i % 3 - 1) * scale;
        return t;
    }

    // ── PerirhinalCortex ────────────────────────────────────────────────────────

    [Fact]
    public void PerirhinalCortex_SignalsNoveltyThenFamiliarity()
    {
        using var per = new PerirhinalCortex(new PerirhinalCortexConfig
        {
            Encoder          = new NetworkConfig { LayerSizes = [8, 12, 6], WeightLearningRate = 0.05f },
            Capacity         = 16,
            NoveltyThreshold = 0.9f,
            BasePath         = Path.Combine(_tmp, "per"),
        }, new Random(1));

        var item  = Vec(1, 0, 1, 0, 1, 0, 1, 0);
        var other = Vec(0, 1, 0, 1, 0, 1, 0, 1);

        Assert.Equal(8, per.ItemDim);
        Assert.Equal(6, per.LatentDim);

        // Nothing remembered yet ⇒ novel, zero familiarity.
        Assert.True(per.IsNovel(item, 10));
        Assert.Equal(0f, per.Familiarity(item, 10), 3);

        for (int i = 0; i < 20; i++) per.Learn(item, 10);
        per.Remember(item, 10);
        Assert.Equal(1, per.MemoryCount);

        // The remembered item is now highly familiar; a different item is less so.
        Assert.True(per.Familiarity(item, 10) > 0.9f);
        Assert.False(per.IsNovel(item, 10));
        Assert.True(per.Familiarity(item, 10) >= per.Familiarity(other, 10));
    }

    // ── ParahippocampalCortex ───────────────────────────────────────────────────

    [Fact]
    public void ParahippocampalCortex_EncodesSceneGist_AndRecognisesPlace()
    {
        using var ph = new ParahippocampalCortex(new ParahippocampalCortexConfig
        {
            Scene = new VisualCortexConfig
            {
                ImageChannels = 1, ImageHeight = 4, ImageWidth = 4,
                PatchSize = 2, PatchStride = 2,
                V1Column    = new NetworkConfig { LayerSizes = [4, 8, 4] },
                HigherAreas = new HierarchicalNetworkConfig
                {
                    Modules = [new NetworkConfig { LayerSizes = [2 * 2 * 4, 8] }, new NetworkConfig { LayerSizes = [8, 4] }],
                },
            },
            Context  = new NetworkConfig { LayerSizes = [4, 2], WeightLearningRate = 0.05f },
            BasePath = Path.Combine(_tmp, "ph"),
        }, new Random(2));

        Assert.Equal(4, ph.SceneLatentDim);

        var img = SceneImg(0.5f);
        Assert.Equal(4, ph.EncodeScene(img, 8).Length);
        ph.Learn(img, Vec(1, 0), 8);
        Assert.InRange(ph.RecognizePlace(img, 8), 0, 1);
    }
}
