using PredictiveCoding.Core;
using PredictiveCoding.Cortex;
using PredictiveCoding.Expanded.DeepRecurrent;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

// Group B — Limbic: Amygdala, OrbitofrontalCortex, CingulateCortex, InsularCortex, Hypothalamus.
public class CortexLimbicTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"limbic_{Guid.NewGuid():N}");

    public void Dispose() { if (Directory.Exists(_tmp)) Directory.Delete(_tmp, recursive: true); }

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);
    private static float Err(DenseTensor<float> a, DenseTensor<float> b)
        => 0.5f * TensorUtils.SumSquares(TensorUtils.ElementwiseSub(a, b));

    // ── Amygdala ────────────────────────────────────────────────────────────────

    [Fact]
    public void Amygdala_AppraisesShape_LearnsAffect_AndReportsSalience()
    {
        using var amy = new Amygdala(new AmygdalaConfig
        {
            AffectDim = 2,
            Appraisal = new NetworkConfig { LayerSizes = [4, 8, 2], WeightLearningRate = 0.05f },
            BasePath  = Path.Combine(_tmp, "amy"),
        }, new Random(1));

        var stim   = Vec(0.5f, -0.3f, 0.1f, 0.7f);
        var affect = Vec(1f, 0f);

        Assert.Equal(2, amy.Appraise(stim, 10).Length);
        float before = Err(amy.Appraise(stim, 10), affect);
        for (int i = 0; i < 60; i++) amy.Learn(stim, affect, 10);
        Assert.True(Err(amy.Appraise(stim, 10), affect) < before);
        Assert.True(amy.SalienceGain(stim, 10, baseGain: 1f) >= 1f);
    }

    [Fact]
    public void Amygdala_RejectsAffectDimMismatch()
        => Assert.Throws<InvalidOperationException>(() => new Amygdala(new AmygdalaConfig
        {
            AffectDim = 3,
            Appraisal = new NetworkConfig { LayerSizes = [4, 2] },
            BasePath  = Path.Combine(_tmp, "bad"),
        }, new Random(1)));

    // ── OrbitofrontalCortex ─────────────────────────────────────────────────────

    [Fact]
    public void OrbitofrontalCortex_LearnsExpectedValue()
    {
        using var ofc = new OrbitofrontalCortex(new OrbitofrontalCortexConfig
        {
            Value    = new NetworkConfig { LayerSizes = [3, 8, 1], WeightLearningRate = 0.05f },
            BasePath = Path.Combine(_tmp, "ofc"),
        }, new Random(2));

        var s = Vec(0.2f, 0.4f, -0.1f);
        float before = MathF.Abs(ofc.EstimateValue(s, 10) - 1f);
        for (int i = 0; i < 80; i++) ofc.Learn(s, 1f, 10);
        Assert.True(MathF.Abs(ofc.EstimateValue(s, 10) - 1f) < before);
    }

    // ── CingulateCortex (ACC) ───────────────────────────────────────────────────

    [Fact]
    public void CingulateCortex_ConflictHighWhenOptionsTie_LowWhenOneDominates()
    {
        var acc = new CingulateCortex(temperature: 1f);
        Assert.Equal(6f, acc.Surprise([1f, 2f, 3f]), 3);
        Assert.Equal(1f, acc.Conflict([2f, 2f]), 3);             // two equal options ⇒ max entropy = 1
        Assert.True(acc.Conflict([1f, 1f, 1f]) > acc.Conflict([0f, 5f, 5f]));
        Assert.True(acc.NeedsControl([1f, 1f, 1f], conflictThreshold: 0.9f));
        Assert.False(acc.NeedsControl([0f, 10f], conflictThreshold: 0.9f));
    }

    // ── InsularCortex ───────────────────────────────────────────────────────────

    [Fact]
    public void InsularCortex_IntegratesInteroceptiveStream()
    {
        using var ins = new InsularCortex(new InsularCortexConfig
        {
            Interoceptive = new DeepRecurrentNetworkConfig
            {
                Stages = [new NetworkConfig { LayerSizes = [4, 8, 6] }, new NetworkConfig { LayerSizes = [6, 8, 4] }],
            },
            BasePath = Path.Combine(_tmp, "insula"),
        }, new Random(3));

        ins.Reset();
        var body = Vec(0.1f, 0.2f, 0.3f, 0.4f);
        Assert.Equal(4, ins.Experience(body, 8).Length);   // top latent dim = Stages[^1].LayerSizes[^1]
        Assert.Equal(4, ins.Feel(body, 8).Length);
    }

    // ── Hypothalamus ────────────────────────────────────────────────────────────

    [Fact]
    public void Hypothalamus_DriveIsSetpointMinusCurrent_AndNeedZeroAtSetpoint()
    {
        var hyp = new Hypothalamus([1f, 0.5f, 0f]);
        Assert.Equal(3, hyp.Dim);

        var drive = hyp.Drive([0.5f, 0.5f, -0.5f]);   // set-point − current = [0.5, 0, 0.5]
        Assert.Equal(0.5f, drive[0], 3);
        Assert.Equal(0f,   drive[1], 3);
        Assert.Equal(0.5f, drive[2], 3);

        Assert.Equal(0f, hyp.NeedLevel([1f, 0.5f, 0f]), 3);   // at set-point ⇒ no need
        Assert.Equal(3, hyp.Goal().Length);
    }
}
