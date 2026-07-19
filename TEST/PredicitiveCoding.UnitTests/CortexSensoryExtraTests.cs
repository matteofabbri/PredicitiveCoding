using PredictiveCoding.Core;
using PredictiveCoding.Cortex;
using PredictiveCoding.Networks.DeepRecurrent;
using PredictiveCoding.Networks.Hopfield;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

// Group D — extra sensory: OlfactoryBulb, GustatoryCortex, VestibularSystem, FrontalEyeFields.
public class CortexSensoryExtraTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"sensory_{Guid.NewGuid():N}");

    public void Dispose() { if (Directory.Exists(_tmp)) Directory.Delete(_tmp, recursive: true); }

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    // ── OlfactoryBulb ───────────────────────────────────────────────────────────

    [Fact]
    public void OlfactoryBulb_WhitensToZeroMean_AndCancelsConstant()
    {
        var bulb = new OlfactoryBulb(receptors: 6);

        var flat = Vec(0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f);
        var w    = bulb.Process(flat);
        Assert.Equal(6, w.Length);
        Assert.All(w.Buffer, v => Assert.True(MathF.Abs(v) < 1e-3f));   // constant ⇒ ≈0

        var varied = Vec(1, 2, 3, 4, 5, 6);
        var wv     = bulb.Process(varied);
        float mean = 0f; foreach (var x in wv.Buffer) mean += x; mean /= wv.Length;
        Assert.True(MathF.Abs(mean) < 1e-3f);                          // zero mean

        Assert.Throws<ArgumentException>(() => bulb.Process(Vec(1, 2, 3)));
    }

    // ── GustatoryCortex ─────────────────────────────────────────────────────────

    [Fact]
    public void GustatoryCortex_MemorisesAndRecognises()
    {
        using var gus = new GustatoryCortex(new GustatoryCortexConfig
        {
            Memory   = new HopfieldNetworkConfig { NetworkConfig = new NetworkConfig { LayerSizes = [12, 16, 8] } },
            BasePath = Path.Combine(_tmp, "gus"),
        }, new Random(1));

        var taste = Vec(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0);
        gus.Learn(taste, 8);
        gus.Memorize(taste, 8);
        Assert.Equal(1, gus.MemoryCount);
        Assert.Equal(12, gus.Recognize(taste, 8).Length);
    }

    // ── VestibularSystem ────────────────────────────────────────────────────────

    [Fact]
    public void VestibularSystem_IntegratesMotionToHeadDirection()
    {
        using var ves = new VestibularSystem(new VestibularSystemConfig
        {
            Integrator = new DeepRecurrentNetworkConfig
            {
                Stages = [new NetworkConfig { LayerSizes = [3, 8, 6] }, new NetworkConfig { LayerSizes = [6, 8, 4] }],
            },
            BasePath = Path.Combine(_tmp, "ves"),
        }, new Random(2));

        ves.Reset();
        var motion = Vec(0.1f, -0.2f, 0.05f);
        Assert.Equal(3, ves.MotionDim);
        Assert.Equal(4, ves.HeadDirectionDim);
        Assert.Equal(4, ves.Integrate(motion, 8).Length);
        Assert.Equal(4, ves.HeadDirection(motion, 8).Length);
    }

    // ── FrontalEyeFields ────────────────────────────────────────────────────────

    [Fact]
    public void FrontalEyeFields_FixatesMostSalientPatch_WithInhibitionOfReturn()
    {
        var fef = new FrontalEyeFields(channels: 1, height: 8, width: 8, patchSize: 4, stride: 4);  // 2×2 = 4 patches
        Assert.Equal(4, fef.PatchCount);

        // High-contrast checkerboard only in the top-left patch; flat elsewhere.
        var img = new DenseTensor<float>(1, 8, 8);
        for (int y = 0; y < 4; y++)
        for (int x = 0; x < 4; x++)
            img[0, y, x] = ((x + y) % 2 == 0) ? 1f : -1f;

        Assert.Equal(4, fef.SaliencyMap(img).Length);
        Assert.Equal(0, fef.NextSaccade(img, new HashSet<int>()));            // top-left is most salient

        var path = fef.Scan(img, 4);
        Assert.Equal(4, path.Count);
        Assert.Equal(4, path.Distinct().Count());                             // inhibition of return ⇒ all distinct
        Assert.Equal(0, path[0]);
        Assert.NotEqual(0, fef.NextSaccade(img, new HashSet<int> { 0 }));     // 0 suppressed ⇒ look elsewhere
    }
}
