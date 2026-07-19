using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Model;
using PredictiveCoding.LLM.Training;

namespace PredicitiveCoding.UnitTests;

// Phase 5 of PredictiveCoding.LLM/ROADMAP.md: hybrid attention + temporal-state
// blocks — the gated diagonal recurrence (RecurrentScan) on the trainable side,
// the O(1)-cache RecurrentTemporalLayer on the inference side.
public class LlmHybridTests
{
    // ── The scan op earns its gradients ──────────────────────────────────────

    [Fact]
    public void RecurrentScan_WithGates_PassesNumericGradientCheck()
    {
        const int T = 4, D = 3;
        int[] targets = [0, 2, 1, 0];
        var rng  = new Random(11);
        float[] aRaw = new float[T * D], iRaw = new float[T * D], uRaw = new float[T * D];
        for (int k = 0; k < T * D; k++)
        {
            aRaw[k] = (float)(rng.NextDouble() * 2 - 1);
            iRaw[k] = (float)(rng.NextDouble() * 2 - 1);
            uRaw[k] = (float)(rng.NextDouble() * 2 - 1);
        }

        // loss = CE( scan( σ(a), σ(i) ⊙ u ), targets ) — exercises Sigmoid, Mul
        // and RecurrentScan in one graph.
        (Tensor a, Tensor i, Tensor u, Tensor loss) Build()
        {
            var a = Tensor.From(aRaw, T, D);
            var i = Tensor.From(iRaw, T, D);
            var u = Tensor.From(uRaw, T, D);
            var s = Ag.RecurrentScan(Ag.Sigmoid(a), Ag.Mul(Ag.Sigmoid(i), u));
            return (a, i, u, Ag.CrossEntropy(s, targets));
        }

        var (ta, ti, tu, loss) = Build();
        loss.Backward();

        foreach (var (raw, grads, name) in new[] { (aRaw, ta.Grad, "a"), (iRaw, ti.Grad, "i"), (uRaw, tu.Grad, "u") })
        {
            for (int k = 0; k < raw.Length; k += 4)                    // sample every 4th index
            {
                const float eps = 1e-3f;
                float keep = raw[k];
                raw[k] = keep + eps; float up   = Build().loss.Data[0];
                raw[k] = keep - eps; float down = Build().loss.Data[0];
                raw[k] = keep;

                float numeric = (up - down) / (2 * eps);
                Assert.True(MathF.Abs(numeric - grads[k]) <= 1e-2f * MathF.Max(1f, MathF.Abs(numeric)),
                    $"{name}[{k}]: numeric {numeric:F4} vs autograd {grads[k]:F4}");
            }
        }
    }

    // ── Pattern plumbing: all-'A' is exactly the old model ───────────────────

    [Fact]
    public void PatternAllA_IsBitIdenticalToNoPattern()
    {
        var baseCfg = new DecoderLmConfig(31, 16, NumBlocks: 2, NumHeads: 2, FfDim: 32, MaxSeq: 8);
        var a = new DecoderLM(baseCfg, new Random(5));
        var b = new DecoderLM(baseCfg with { BlockPattern = "AA" }, new Random(5));

        Assert.Equal(a.Parameters().Length, b.Parameters().Length);
        var la = a.Forward([1, 2, 3]);
        var lb = b.Forward([1, 2, 3]);
        for (int k = 0; k < la.Data.Length; k++)
            Assert.Equal(la.Data[k], lb.Data[k]);
    }

    [Fact]
    public void BlockPattern_Validation_Throws()
    {
        var cfg = new DecoderLmConfig(31, 16, NumBlocks: 2, NumHeads: 2, FfDim: 32, MaxSeq: 8);
        Assert.Throws<ArgumentException>(() => (cfg with { BlockPattern = "A" }).Validate());     // wrong length
        Assert.Throws<ArgumentException>(() => (cfg with { BlockPattern = "AX" }).Validate());    // wrong char

        var gpt = new TransformerConfig { VocabSize = 260, MaxSeqLen = 8, EmbedDim = 16, NumHeads = 2, NumLayers = 2, FeedForwardDim = 32 };
        Assert.Throws<InvalidOperationException>(() => (gpt with { BlockPattern = "T" }).Validate());
        Assert.Throws<InvalidOperationException>(() => (gpt with { BlockPattern = "TZ" }).Validate());
    }

    // ── The state actually carries the past ──────────────────────────────────

    [Fact]
    public void PureTemporalModel_LearnsToEchoThePreviousToken()
    {
        // Target[t] = tokens[t−1]: with no attention at all, the only way to
        // answer is to carry the previous token through the recurrent state.
        var lm  = new DecoderLM(new DecoderLmConfig(13, 16, NumBlocks: 2, NumHeads: 1, FfDim: 32, MaxSeq: 8, BlockPattern: "TT"), new Random(4));
        var opt = new Adam();
        int[] toks = [5, 2, 7, 3, 9, 4], tgts = [-1, 5, 2, 7, 3, 9];

        float first = lm.Loss(toks, tgts, ignoreIndex: -1).Data[0];
        for (int it = 0; it < 250; it++)
        {
            lm.Loss(toks, tgts, ignoreIndex: -1, training: true).Backward();
            opt.Step(lm.Parameters(), 0.01f);
        }
        float last = lm.Loss(toks, tgts, ignoreIndex: -1).Data[0];

        Assert.True(last < first * 0.4f, $"the temporal state did not learn the echo task: {first:F3} → {last:F3}");
    }

    // ── Inference twin: O(1) state vs the growing transcript ─────────────────

    [Fact]
    public void RecurrentLayer_CacheStaysConstant_WhileAttentionGrows()
    {
        var cfg = new TransformerConfig
        {
            VocabSize = 260, MaxSeqLen = 32, EmbedDim = 16, NumHeads = 2, NumLayers = 2,
            FeedForwardDim = 32, BlockPattern = "TA",
        };
        var model = new GptModel(cfg, new Random(7));
        var cache = model.CreateCache();

        model.Step(65, 0, cache);
        long temporalAfter1 = cache.Layers[0].BytesUsed;
        long kvAfter1       = cache.Layers[1].BytesUsed;

        for (int t = 1; t < 20; t++)
        {
            var logits = model.Step(65 + t, t, cache);
            Assert.All(logits.Buffer, f => Assert.True(float.IsFinite(f)));
        }

        Assert.Equal(temporalAfter1, cache.Layers[0].BytesUsed);       // O(1): one state vector, forever
        Assert.Equal(16 * sizeof(float), temporalAfter1);
        Assert.Equal(kvAfter1 * 20, cache.Layers[1].BytesUsed);        // O(n): the transcript grows per step
        Assert.Equal(20, cache.Layers[0].Length);                      // steps absorbed
    }

    [Fact]
    public void NemotronStylePreset_ValidatesAndRuns()
    {
        var cfg = ModernPresets.NemotronStyle(embed: 16, heads: 2, kvHeads: 1, layers: 4, ff: 32, maxSeq: 16);
        Assert.Equal("TTTA", cfg.BlockPattern);
        cfg.Validate();

        var model = new GptModel(cfg, new Random(9));
        var cache = model.CreateCache();
        for (int t = 0; t < 5; t++)
        {
            var logits = model.Step(65 + t, t, cache);
            Assert.All(logits.Buffer, f => Assert.True(float.IsFinite(f)));
        }

        // Three constant-size temporal layers + one growing KV layer.
        Assert.Equal(16 * sizeof(float), cache.Layers[0].BytesUsed);
        Assert.Equal(16 * sizeof(float), cache.Layers[1].BytesUsed);
        Assert.Equal(16 * sizeof(float), cache.Layers[2].BytesUsed);
        Assert.True(cache.Layers[3].BytesUsed > cache.Layers[0].BytesUsed);
    }
}
