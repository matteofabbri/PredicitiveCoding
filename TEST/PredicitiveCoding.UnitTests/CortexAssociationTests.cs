using PredictiveCoding.Core;
using PredictiveCoding.Cortex;
using PredictiveCoding.Networks.MultiModal;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

// Group G — association / social / categorical: TemporoparietalJunction, FrontopolarCortex,
// SuperiorTemporalSulcus, FusiformFaceArea, VisualWordFormArea.
public class CortexAssociationTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"assoc_{Guid.NewGuid():N}");

    public void Dispose() { if (Directory.Exists(_tmp)) Directory.Delete(_tmp, recursive: true); }

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);
    private static float Err(DenseTensor<float> a, DenseTensor<float> b)
        => 0.5f * TensorUtils.SumSquares(TensorUtils.ElementwiseSub(a, b));

    // ── TemporoparietalJunction ─────────────────────────────────────────────────

    [Fact]
    public void TemporoparietalJunction_ReorientsOnSurprise_NotWhenFocused()
    {
        var tpj = new TemporoparietalJunction(reorientThreshold: 1.5f);

        Assert.True(tpj.ShouldReorient(bottomUpSalience: 2f, topDownFocus: 0.5f));   // 2 > 1.5·0.5
        Assert.False(tpj.ShouldReorient(bottomUpSalience: 0.5f, topDownFocus: 1f));  // 0.5 < 1.5
        Assert.Equal(0f, tpj.ReorientSignal(0.5f, 1f), 3);
        Assert.True(tpj.ReorientSignal(2f, 0.5f) > 0f);
    }

    // ── FrontopolarCortex ───────────────────────────────────────────────────────

    [Fact]
    public void FrontopolarCortex_SwitchesToABetterHeldGoal_WithMargin()
    {
        var fp = new FrontopolarCortex(switchMargin: 0.1f);

        Assert.False(fp.HasPending);
        Assert.False(fp.ShouldSwitch(activeGoalValue: 1f));   // nothing held

        fp.Hold(0.8f);
        Assert.True(fp.HasPending);
        Assert.False(fp.ShouldSwitch(1f));                    // 0.8 not > 1 + 0.1

        fp.Hold(1.5f);
        Assert.True(fp.ShouldSwitch(1f));                     // 1.5 > 1.1

        fp.Clear();
        Assert.False(fp.HasPending);
    }

    // ── SuperiorTemporalSulcus ──────────────────────────────────────────────────

    [Fact]
    public void SuperiorTemporalSulcus_BindsModalities_AndClassifies()
    {
        using var sts = new SuperiorTemporalSulcus(new SuperiorTemporalSulcusConfig
        {
            MultiModal = new MultiModalNetworkConfig
            {
                Modalities =
                [
                    new ModalityConfig { Name = "motion", NetworkConfig = new NetworkConfig { LayerSizes = [4, 4] } },
                    new ModalityConfig { Name = "voice",  NetworkConfig = new NetworkConfig { LayerSizes = [3, 4] } },
                ],
                AssociativeConfig = new NetworkConfig { LayerSizes = [8, 3] },   // 4+4 latents → 3 agents
            },
            BasePath = Path.Combine(_tmp, "sts"),
        }, new Random(1));

        var streams = new Dictionary<int, DenseTensor<float>> { [0] = Vec(1, 0, 1, 0), [1] = Vec(0, 1, 0) };
        Assert.Equal(2, sts.ModalityCount);
        Assert.Equal(8, sts.Integrate(streams, 8).Length);
        sts.Learn(streams, Vec(1, 0, 0), 8);
        Assert.InRange(sts.Classify(streams, 8), 0, 2);
    }

    // ── FusiformFaceArea ────────────────────────────────────────────────────────

    [Fact]
    public void FusiformFaceArea_LearnsFaceEmbedding()
    {
        using var ffa = new FusiformFaceArea(new FusiformFaceAreaConfig
        {
            FaceLatentDim = 3,
            Recognition = new NetworkConfig { LayerSizes = [6, 8, 3], WeightLearningRate = 0.05f },
            BasePath = Path.Combine(_tmp, "ffa"),
        }, new Random(2));

        var visual = Vec(0.2f, 0.5f, -0.1f, 0.3f, 0.0f, 0.4f);
        var face = Vec(1f, 0f, 0f);

        Assert.Equal(3, ffa.Encode(visual, 10).Length);
        float before = Err(ffa.Encode(visual, 10), face);
        for (int i = 0; i < 60; i++) ffa.Learn(visual, face, 10);
        Assert.True(Err(ffa.Encode(visual, 10), face) < before);
    }

    [Fact]
    public void FusiformFaceArea_RejectsDimMismatch()
        => Assert.Throws<InvalidOperationException>(() => new FusiformFaceArea(new FusiformFaceAreaConfig
        {
            FaceLatentDim = 4,
            Recognition = new NetworkConfig { LayerSizes = [6, 2] },
            BasePath = Path.Combine(_tmp, "bad"),
        }, new Random(1)));

    // ── VisualWordFormArea ──────────────────────────────────────────────────────

    [Fact]
    public void VisualWordFormArea_LearnsOrthographicCode()
    {
        using var vwfa = new VisualWordFormArea(new VisualWordFormAreaConfig
        {
            CodeDim = 4,
            Orthography = new NetworkConfig { LayerSizes = [8, 10, 4], WeightLearningRate = 0.05f },
            BasePath = Path.Combine(_tmp, "vwfa"),
        }, new Random(3));

        var word = Vec(1, 0, 1, 0, 1, 0, 1, 0);
        var code = Vec(1, 0, 0, 0);

        Assert.Equal(4, vwfa.Recognize(word, 10).Length);
        float before = Err(vwfa.Recognize(word, 10), code);
        for (int i = 0; i < 60; i++) vwfa.Learn(word, code, 10);
        Assert.True(Err(vwfa.Recognize(word, 10), code) < before);
    }
}
