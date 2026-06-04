using PredictiveCoding.LLM.Generation;
using PredictiveCoding.LLM.Model;
using PredictiveCoding.LLM.Tokenization;
using PredictiveCoding.LLM.Training;

namespace PredicitiveCoding.UnitTests;

// Modern (2024-2025) architecture features: SwiGLU, QK-Norm, attention/final
// logit soft-capping, and Multi-head Latent Attention (MLA / DeepSeek-V3).
public class LlmModernTests
{
    private static DenseTensorLogits Step(TransformerConfig cfg)
    {
        var model  = new GptModel(cfg, new Random(7));
        var logits = model.Step(65, 0, model.CreateCache());
        return new DenseTensorLogits(model.VocabSize, logits.Buffer);
    }

    private readonly record struct DenseTensorLogits(int VocabSize, float[] Buffer);

    [Fact]
    public void SwiGlu_ProducesFiniteLogits()
    {
        var r = Step(new TransformerConfig { VocabSize = 260, MaxSeqLen = 16, EmbedDim = 32, NumHeads = 4, NumLayers = 2, FeedForwardDim = 64, Ffn = FeedForwardKind.SwiGlu });
        Assert.Equal(260, r.Buffer.Length);
        Assert.All(r.Buffer, f => Assert.True(float.IsFinite(f)));
    }

    [Fact]
    public void QkNorm_ProducesFiniteLogits()
    {
        var r = Step(new TransformerConfig { VocabSize = 260, MaxSeqLen = 16, EmbedDim = 32, NumHeads = 4, NumLayers = 2, FeedForwardDim = 64, UseQkNorm = true, Positional = PositionalKind.Rotary });
        Assert.All(r.Buffer, f => Assert.True(float.IsFinite(f)));
    }

    [Fact]
    public void SoftCapping_BoundsFinalLogits()
    {
        var cfg = new TransformerConfig { VocabSize = 260, MaxSeqLen = 16, EmbedDim = 32, NumHeads = 4, NumLayers = 2, FeedForwardDim = 64, AttnLogitSoftcap = 20f, FinalLogitSoftcap = 10f };
        var r = Step(cfg);
        Assert.All(r.Buffer, f => Assert.InRange(f, -10f, 10f));   // capped by FinalLogitSoftcap
    }

    [Fact]
    public void Mla_ProducesFiniteLogits_AndCachesLatent()
    {
        var cfg = new TransformerConfig
        {
            VocabSize = 260, MaxSeqLen = 16, EmbedDim = 32, NumHeads = 4, NumLayers = 2, FeedForwardDim = 64,
            Attention = AttentionKind.Latent, KvLatentDim = 16, RopeHeadDim = 8,
        };
        var model = new GptModel(cfg, new Random(3));
        var cache = model.CreateCache();
        var l0 = model.Step(10, 0, cache);
        var l1 = model.Step(20, 1, cache);
        Assert.Equal(2, cache.Length);                  // latent cache grew per step
        Assert.All(l0.Buffer, f => Assert.True(float.IsFinite(f)));
        Assert.All(l1.Buffer, f => Assert.True(float.IsFinite(f)));
    }

    [Fact]
    public void DeepSeekPreset_GeneratesAndEvaluates()
    {
        var model = new GptModel(ModernPresets.DeepSeekStyle(maxSeq: 32, embed: 32, heads: 4, layers: 2, ff: 64), new Random(1));
        var tok   = new ByteTokenizer();

        var gen = new TextGenerator(model, tok)
            .GenerateTokens(tok.Encode("hi", addBos: true), new SamplingOptions { MaxNewTokens = 8, Greedy = true });
        Assert.True(gen.Count <= 8);
        Assert.All(gen, t => Assert.InRange(t, 0, model.VocabSize - 1));

        var (loss, ppl) = CrossEntropy.Evaluate(model, tok.Encode("predictive"));
        Assert.True(float.IsFinite(loss) && ppl >= 1f);
    }

    [Fact]
    public void Llama3Preset_ProducesFiniteLogits()
    {
        var r = Step(ModernPresets.Llama3Style(maxSeq: 16, embed: 32, heads: 4, kvHeads: 2, layers: 2, ff: 64));
        Assert.All(r.Buffer, f => Assert.True(float.IsFinite(f)));
    }
}
