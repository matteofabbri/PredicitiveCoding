using PredictiveCoding.Core;
using PredictiveCoding.Cortex;
using PredictiveCoding.Expanded.MultiModal;
using PredictiveCoding.Expanded.Networks;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

// Group F — motor / integration: PosteriorParietalCortex, PremotorCortex, Claustrum.
public class CortexIntegrationTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"integration_{Guid.NewGuid():N}");

    public void Dispose() { if (Directory.Exists(_tmp)) Directory.Delete(_tmp, recursive: true); }

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    // ── PosteriorParietalCortex ─────────────────────────────────────────────────

    [Fact]
    public void PosteriorParietalCortex_TransformsFrames_AndLocatesTarget()
    {
        using var ppc = new PosteriorParietalCortex(new PosteriorParietalCortexConfig
        {
            MultiModal = new MultiModalNetworkConfig
            {
                Modalities =
                [
                    new ModalityConfig { Name = "retinal", NetworkConfig = new NetworkConfig { LayerSizes = [4, 4] } },
                    new ModalityConfig { Name = "eye",     NetworkConfig = new NetworkConfig { LayerSizes = [2, 4] } },
                ],
                AssociativeConfig = new NetworkConfig { LayerSizes = [8, 3] },   // 4+4 latents → 3 target zones
            },
            BasePath = Path.Combine(_tmp, "ppc"),
        }, new Random(1));

        var frames = new Dictionary<int, DenseTensor<float>> { [0] = Vec(1, 0, 1, 0), [1] = Vec(0, 1) };
        Assert.Equal(8, ppc.Transform(frames, 8).Length);              // joint latent = sum of modality latents
        ppc.Learn(frames, Vec(1, 0, 0), 8);
        Assert.InRange(ppc.Locate(frames, 8), 0, 2);
    }

    // ── PremotorCortex (SMA) ────────────────────────────────────────────────────

    [Fact]
    public void PremotorCortex_LearnsSequence_AndPredictsNextAction()
    {
        using var pre = new PremotorCortex(new PremotorCortexConfig
        {
            Sequencer = new TemporalMoEConfig
            {
                ExpertCount              = 2,
                ExpertConfig             = new NetworkConfig { LayerSizes = [3, 8, 2] },   // state 3 → 2 actions
                DefaultTemporalPrecision = 1f,
                GatingTemperature        = 1f,
            },
            BasePath = Path.Combine(_tmp, "pre"),
        }, new Random(2));

        Assert.Equal(2, pre.ExpertCount);
        pre.Reset();
        var st = Vec(0.2f, -0.1f, 0.4f);
        pre.LearnStep(st, Vec(1, 0), 6);
        pre.LearnStep(Vec(0.1f, 0.3f, -0.2f), Vec(0, 1), 6);
        Assert.InRange(pre.NextAction(st, 6), 0, 1);
    }

    // ── Claustrum ───────────────────────────────────────────────────────────────

    [Fact]
    public void Claustrum_BroadcastsConsensus_AndScoresCoherence()
    {
        var cl = new Claustrum();
        var a  = Vec(1, 0, 0);
        var b  = Vec(1, 0, 0);
        var c  = Vec(0, 1, 0);

        var mean = cl.Synchronize([a, b]);
        Assert.Equal(3, mean.Length);
        Assert.Equal(1f, mean.Buffer[0], 3);
        Assert.Equal(0f, mean.Buffer[1], 3);

        Assert.Equal(1f, cl.Coherence([a, b]), 3);                     // identical ⇒ fully coherent
        Assert.Equal(0f, cl.Coherence([a, c]), 3);                     // orthogonal ⇒ no coherence
        Assert.True(cl.Coherence([a, b]) > cl.Coherence([a, c]));
        Assert.Equal(1f, cl.Coherence([a]), 3);                        // single area ⇒ 1

        Assert.Throws<ArgumentException>(() => cl.Synchronize([Vec(1, 0), Vec(1, 0, 0)]));
    }
}
