using PredictiveCoding.Core;
using PredictiveCoding.Cortex;
using PredictiveCoding.Networks.DeepRecurrent;
using PredictiveCoding.Networks.Generative;
using PredictiveCoding.Networks.Graph;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

// Group C — Language: WernickeArea, BrocaArea, AngularGyrus.
public class CortexLanguageTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"language_{Guid.NewGuid():N}");

    public void Dispose() { if (Directory.Exists(_tmp)) Directory.Delete(_tmp, recursive: true); }

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    // ── WernickeArea (comprehension) ────────────────────────────────────────────

    [Fact]
    public void WernickeArea_ComprehendsTokenSequenceToSemanticLatent()
    {
        using var wer = new WernickeArea(new WernickeAreaConfig
        {
            Sequence = new DeepRecurrentNetworkConfig
            {
                Stages = [new NetworkConfig { LayerSizes = [4, 8, 6] }, new NetworkConfig { LayerSizes = [6, 8, 4] }],
            },
            BasePath = Path.Combine(_tmp, "wer"),
        }, new Random(1));

        Assert.Equal(4, wer.TokenDim);
        Assert.Equal(4, wer.SemanticDim);

        var tokens = new[] { Vec(1, 0, 0, 0), Vec(0, 1, 0, 0), Vec(0, 0, 1, 0) };
        Assert.Equal(4, wer.LearnUtterance(tokens, 8).Length);   // semantic dim = TopLatentDim
        Assert.Equal(4, wer.Comprehend(tokens, 8).Length);
    }

    // ── BrocaArea (production) ───────────────────────────────────────────────────

    [Fact]
    public void BrocaArea_ProducesUtteranceFromSemanticLatent()
    {
        using var bro = new BrocaArea(new BrocaAreaConfig
        {
            Generative = new GenerativeNetworkConfig { NetworkConfig = new NetworkConfig { LayerSizes = [6, 8, 3], WeightLearningRate = 0.05f } },
            DriftScale = 0.2f,
            BasePath   = Path.Combine(_tmp, "bro"),
        }, new Random(2));

        Assert.Equal(3, bro.SemanticDim);
        Assert.Equal(6, bro.UtteranceDim);

        var utter = Vec(0.2f, 0.4f, 0.1f, 0.5f, 0.3f, 0.6f);
        for (int i = 0; i < 10; i++) bro.LearnUtterance(utter, 8);

        var semantic = Vec(0.1f, -0.2f, 0.3f);
        Assert.Equal(6, bro.Produce(semantic, 8).Length);

        var seq = bro.ProduceSequence(semantic, length: 4, steps: 8, new Random(7));
        Assert.Equal(4, seq.Count);
        Assert.All(seq, u => Assert.Equal(6, u.Length));

        Assert.Throws<ArgumentException>(() => bro.Produce(Vec(1f, 2f), 8));   // wrong semantic dim
    }

    // ── AngularGyrus (relational semantic hub) ──────────────────────────────────

    [Fact]
    public void AngularGyrus_LearnsRelations_AndCompletesConcepts()
    {
        using var ang = new AngularGyrus(new AngularGyrusConfig
        {
            Relations = new GraphPredictiveNetworkConfig
            {
                EdgePredictorConfig = new NetworkConfig { LayerSizes = [4, 8, 4], WeightLearningRate = 0.05f },
                BlendRate           = 0.5f,
            },
            BasePath = Path.Combine(_tmp, "ang"),
        }, new Random(3));

        Assert.Equal(4, ang.ConceptDim);

        var concepts  = new[] { Vec(1, 0, 0, 0), Vec(0, 1, 0, 0), Vec(0, 0, 1, 0) };
        var relations = new[] { new[] { 1 }, new[] { 0, 2 }, new[] { 1 } };   // chain 0–1–2

        float before = ang.RelationalEnergy(concepts, relations, 8);
        for (int i = 0; i < 50; i++) ang.Learn(concepts, relations, 8);
        Assert.True(ang.RelationalEnergy(concepts, relations, 8) < before);

        var noisy     = new[] { concepts[0], Vec(0, 0, 0, 0), concepts[2] };
        var observed  = new[] { true, false, true };
        var completed = ang.Associate(noisy, relations, observed, steps: 8, rounds: 5);
        Assert.Equal(3, completed.Length);
        Assert.Equal(4, completed[1].Length);
    }
}
