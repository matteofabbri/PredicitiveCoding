using PredictiveCoding.Core;
using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Data;
using PredictiveCoding.LLM.Generation;
using PredictiveCoding.LLM.Model;
using PredictiveCoding.LLM.Tokenization;

namespace PredicitiveCoding.UnitTests;

public class LlmFeaturesTests
{
    private static GptModel TinyModel(TransformerConfig? cfg = null) =>
        new(cfg ?? new TransformerConfig
        {
            VocabSize = 260, MaxSeqLen = 32, EmbedDim = 32, NumHeads = 4, NumLayers = 2, FeedForwardDim = 64,
        }, new Random(42));

    // ── 1. BPE tokenizer ─────────────────────────────────────────────────────

    [Fact]
    public void Bpe_TrainsAndRoundTrips()
    {
        var bpe = new BpeTokenizer();
        bpe.Train("the cat sat on the mat. the cat ran after the cat.", numMerges: 30);

        Assert.True(bpe.VocabSize > 260, "should have learned merges");
        var ids = bpe.Encode("the cat", addBos: true, addEos: true);
        Assert.Equal(bpe.Bos, ids[0]);
        Assert.Equal(bpe.Eos, ids[^1]);
        Assert.Equal("the cat", bpe.Decode(ids));
        Assert.Equal("hello world", bpe.Decode(bpe.Encode("hello world")));
        Assert.True(bpe.Encode("the the the").Count < "the the the".Length, "merges should shorten the sequence");
    }

    // ── 2. Streaming dataset ───────────────────────────────────────────────────

    [Fact]
    public void StreamingDataset_YieldsShiftedWindows()
    {
        var ds = new StreamingTextDataset(["abcdef", "ghijkl"], new ByteTokenizer(), seqLen: 4, stride: 2);
        var windows = ds.Windows().ToList();

        Assert.NotEmpty(windows);
        var (input, target) = windows[0];
        Assert.Equal(4, input.Length);
        for (int i = 0; i < 3; i++) Assert.Equal(input[i + 1], target[i]);
    }

    // ── 3-6. RoPE / RMSNorm / GQA / flash ──────────────────────────────────────

    [Fact]
    public void AdvancedAttention_ProducesFiniteLogits()
    {
        var cfg = new TransformerConfig
        {
            VocabSize = 260, MaxSeqLen = 16, EmbedDim = 32, NumHeads = 4, NumKvHeads = 2, NumLayers = 2,
            FeedForwardDim = 64, Positional = PositionalKind.Rotary, Norm = NormKind.RmsNorm, UseFlashAttention = true,
        };
        var model  = new GptModel(cfg, new Random(1));
        var logits = model.Step(65, 0, model.CreateCache());
        Assert.Equal(cfg.VocabSize, logits.Length);
        Assert.All(logits.Buffer, f => Assert.True(float.IsFinite(f)));
    }

    [Fact]
    public void FlashAttention_MatchesStandard()
    {
        var baseCfg  = new TransformerConfig { VocabSize = 260, MaxSeqLen = 16, EmbedDim = 32, NumHeads = 4, NumLayers = 2, FeedForwardDim = 64 };
        var standard = new GptModel(baseCfg with { UseFlashAttention = false }, new Random(7));
        var flash    = new GptModel(baseCfg with { UseFlashAttention = true  }, new Random(7));

        // Same seed ⇒ identical weights; online softmax must match the standard path.
        var ls = standard.Step(40, 0, standard.CreateCache());
        var lf = flash.Step(40, 0, flash.CreateCache());
        for (int j = 0; j < ls.Length; j++)
            Assert.True(MathF.Abs(ls.Buffer[j] - lf.Buffer[j]) < 1e-3f);
    }

    // ── 7. Beam search ─────────────────────────────────────────────────────────

    [Fact]
    public void BeamSearch_IsDeterministicAndInRange()
    {
        var model = TinyModel();
        var tok   = new ByteTokenizer();
        var bs    = new BeamSearch(model, tok);
        var opts  = new BeamSearchOptions { BeamWidth = 3, MaxNewTokens = 8 };

        var a = bs.SearchTokens(tok.Encode("hi", addBos: true), opts);
        var b = bs.SearchTokens(tok.Encode("hi", addBos: true), opts);
        Assert.Equal(a, b);
        Assert.All(a, t => Assert.InRange(t, 0, model.VocabSize - 1));
    }

    // ── 8. Frequency / presence penalties ──────────────────────────────────────

    [Fact]
    public void PresencePenalty_AvoidsAlreadyGeneratedToken()
    {
        var logits = new DenseTensor<float>([2f, 1f, 0f, 0f], 4);  // greedy default → token 0
        var sampler = new Sampler(0);

        Assert.Equal(0, sampler.Sample(logits, Array.Empty<int>(), new SamplingOptions { Greedy = true }));
        int penalized = sampler.Sample(logits, [0], new SamplingOptions { Greedy = true, PresencePenalty = 10f });
        Assert.NotEqual(0, penalized);
    }

    // ── 9. Batched decoding ─────────────────────────────────────────────────────

    [Fact]
    public void BatchedGenerator_DecodesEachSequence()
    {
        var model = TinyModel();
        var tok   = new ByteTokenizer();
        var bg    = new BatchedGenerator(model, tok);

        var outs = bg.GenerateBatch(
            [tok.Encode("a", addBos: true), tok.Encode("bb", addBos: true)],
            new SamplingOptions { MaxNewTokens = 8, Greedy = true });

        Assert.Equal(2, outs.Count);
        Assert.All(outs, o => Assert.True(o.Count <= 8));
    }

    // ── 10. Autodiff (backprop through the transformer) ────────────────────────

    [Fact]
    public void Autodiff_GradientCheck_MatchesNumeric()
    {
        var lm   = new TinyTransformerLM(vocab: 5, dim: 4, ff: 8, new Random(1));
        int[] toks = [1, 2, 3];
        int[] tgts = [2, 3, 4];

        lm.Loss(toks, tgts).Backward();   // fills analytic gradients

        float Loss() => lm.Loss(toks, tgts).Data[0];
        foreach (var p in new[] { lm.Wout, lm.Wq, lm.W1, lm.Ln1G, lm.Wemb, lm.Wo })
            for (int idx = 0; idx < Math.Min(3, p.Data.Length); idx++)
                CheckGrad(Loss, p, idx);
    }

    [Fact]
    public void Autodiff_SgdReducesLoss()
    {
        var lm   = new TinyTransformerLM(vocab: 6, dim: 8, ff: 16, new Random(2));
        int[] toks = [1, 2, 3, 4];
        int[] tgts = [2, 3, 4, 5];

        float first = lm.Loss(toks, tgts).Data[0];
        for (int it = 0; it < 500; it++)
        {
            var loss = lm.Loss(toks, tgts);
            loss.Backward();
            Sgd.Step(lm.Parameters(), 0.3f);
        }
        float last = lm.Loss(toks, tgts).Data[0];
        Assert.True(last < first - 0.3f, $"expected the loss to fall substantially (first={first}, last={last}).");
    }

    private static void CheckGrad(Func<float> loss, Tensor p, int idx)
    {
        float analytic = p.Grad[idx];
        float orig     = p.Data[idx];
        const float eps = 2e-3f;

        p.Data[idx] = orig + eps; float l2 = loss();
        p.Data[idx] = orig - eps; float l1 = loss();
        p.Data[idx] = orig;

        float numeric = (l2 - l1) / (2f * eps);
        Assert.True(MathF.Abs(numeric - analytic) <= 2e-2f * (1f + MathF.Abs(analytic)),
            $"gradient mismatch at {idx}: numeric={numeric}, analytic={analytic}");
    }
}
