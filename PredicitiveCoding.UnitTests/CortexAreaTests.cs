using PredictiveCoding.Core;
using PredictiveCoding.Cortex;
using PredictiveCoding.Expanded.Hierarchical;
using PredictiveCoding.Expanded.Hopfield;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

public class CortexAreaTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"cortex_{Guid.NewGuid():N}");

    public void Dispose() { if (Directory.Exists(_tmp)) Directory.Delete(_tmp, recursive: true); }

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    // ── Pure helpers (no persistence) ──────────────────────────────────────────

    [Fact]
    public void RetinaLgn_DoG_CancelsOnFlatImage_AndKeepsShape()
    {
        var retina = new RetinaLgn(channels: 1, height: 4, width: 4, centerSigma: 0.6f, surroundSigma: 1.5f);
        var flat   = new DenseTensor<float>(1, 4, 4);
        Array.Fill(flat.Buffer, 0.5f);

        var outp = retina.Process(flat);
        Assert.Equal(16, outp.Length);
        Assert.All(outp.Buffer, v => Assert.True(MathF.Abs(v) < 1e-3f));   // center-surround cancels on a constant
    }

    [Fact]
    public void RetinaLgn_RejectsBadSigmas()
        => Assert.Throws<ArgumentException>(() => new RetinaLgn(1, 4, 4, centerSigma: 1.5f, surroundSigma: 0.6f));

    [Fact]
    public void Thalamus_AllocatesMorePrecisionToReliableAreas()
    {
        var thal = new Thalamus(temperature: 1f, basePrecision: 1f);
        var p = thal.Allocate([1f, 5f]);   // area 0 has lower energy ⇒ more reliable

        Assert.Equal(2, p.Length);
        Assert.True(p[0] > p[1]);
        Assert.Equal(2f, p[0] + p[1], 3);  // mean precision == BasePrecision
    }

    [Fact]
    public void BasalGanglia_SelectsLowestEnergy_AndWithholdsAboveThreshold()
    {
        var (action, gates) = new BasalGanglia(gatingTemperature: 1f).Select([2f, 0.5f, 3f]);
        Assert.Equal(1, action);
        Assert.Equal(3, gates.Length);

        var (noGo, _) = new BasalGanglia(1f, noGoThreshold: 0f).Select([2f, 3f]);
        Assert.Equal(-1, noGo);            // best energy 2 > threshold 0 ⇒ No-Go
    }

    // ── Persistence-backed areas ───────────────────────────────────────────────

    [Fact]
    public void CerebellarCortex_PredictsShape_AndLearningReducesError()
    {
        using var cb = new CerebellarCortex(new CerebellarCortexConfig
        {
            StateDim = 3, CommandDim = 2,
            ForwardModel = new NetworkConfig { LayerSizes = [5, 8, 3], WeightLearningRate = 0.05f },
            BasePath = _tmp,
        }, new Random(1));

        var s = Vec(0.2f, -0.1f, 0.3f);
        var c = Vec(0.5f, -0.5f);
        var target = Vec(0.4f, -0.2f, 0.1f);

        Assert.Equal(3, cb.Predict(s, c, 10).Length);
        float before = cb.PredictionError(s, c, target, 10);
        for (int i = 0; i < 80; i++) cb.Learn(s, c, target, 10);
        Assert.True(cb.PredictionError(s, c, target, 10) < before);
    }

    [Fact]
    public void EntorhinalCortex_EncodesGridCode_AndPredictsObservation()
    {
        using var ec = new EntorhinalCortex(new EntorhinalCortexConfig
        {
            GridModules = 3, CellsPerModule = 8,
            Association = new NetworkConfig { LayerSizes = [24, 8] },   // 3·8 = 24 grid cells → 8-D observation
            BasePath = _tmp,
        }, new Random(2));

        Assert.Equal(24, ec.EncodePosition(1.2f, -0.5f).Length);
        ec.Learn(1.2f, -0.5f, Vec(0, 1, 0, 1, 0, 1, 0, 1), 8);
        Assert.Equal(8, ec.PredictObservation(1.2f, -0.5f, 8).Length);
    }

    [Fact]
    public void PiriformCortex_MemorisesAndRecognises()
    {
        using var pir = new PiriformCortex(new PiriformCortexConfig
        {
            Memory   = new HopfieldNetworkConfig { NetworkConfig = new NetworkConfig { LayerSizes = [12, 16, 8] } },
            BasePath = _tmp,
        }, new Random(3));

        var odor = Vec(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0);
        pir.Learn(odor, 8);
        pir.Memorize(odor, 8);
        Assert.Equal(1, pir.MemoryCount);
        Assert.Equal(12, pir.Recognize(odor, 8).Length);
    }

    [Fact]
    public void AuditoryCortex_EncodesAndRecognises()
    {
        using var aud = new AuditoryCortex(new AuditoryCortexConfig
        {
            FrequencyBins = 8, TimeFrames = 8, PatchSize = 4, PatchStride = 4,   // 2×2 grid of patches
            A1Column    = new NetworkConfig { LayerSizes = [16, 8, 4] },          // 1·4·4 = 16 → K = 4
            BeltParabelt = new HierarchicalNetworkConfig { Modules = [new NetworkConfig { LayerSizes = [2 * 2 * 4, 8] }, new NetworkConfig { LayerSizes = [8, 4] }] },
            Classifier  = new NetworkConfig { LayerSizes = [4, 3] },
            BasePath    = _tmp,
        }, new Random(4));

        var spec = new DenseTensor<float>(1, 8, 8);
        for (int i = 0; i < spec.Length; i++) spec.Buffer[i] = (i % 5 - 2) * 0.2f;

        Assert.Equal(4, aud.Encode(spec, 6).Length);
        aud.LearnSupervised(spec, Vec(1, 0, 0), 6);
        Assert.InRange(aud.Recognize(spec, 6), 0, 2);
    }

    [Fact]
    public void HippocampalFormation_StoresRecallsAndConsolidates()
    {
        using var hpc = new HippocampalFormation(new HippocampalFormationConfig
        {
            Hippocampus = new HopfieldNetworkConfig { NetworkConfig = new NetworkConfig { LayerSizes = [16, 12, 8] } },
            Neocortex   = new NetworkConfig { LayerSizes = [16, 10, 8] },
            MaxEpisodes = 8,
            BasePath    = _tmp,
        }, new Random(5));

        var input = Vec(1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0);
        hpc.Experience(input, 8);
        Assert.Equal(1, hpc.EpisodeCount);
        Assert.Equal(16, hpc.Recall(input, 8).Length);

        hpc.Replay(rounds: 2, steps: 8, new Random(6));
        Assert.Equal(8, hpc.NeocorticalEncode(input, 8).Length);
    }
}
