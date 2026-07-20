using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Tokenization;

namespace PredicitiveCoding.UnitTests;

// ROADMAP-Scale Phase B1: real config presets. The embedding becomes a row
// gather (same maths as the one-hot matmul, none of the [seq, vocab] waste),
// and Small/Base presets size the model for real training on one RTX 4070.
public class LlmScaleBTests
{
    // ── Embedding: gather ≡ one-hot matmul ───────────────────────────────────

    [Fact]
    public void EmbeddingGather_MatchesOneHotMatMul_ForwardAndGradient()
    {
        const int vocab = 11, dim = 5;
        int[] tokens = [3, 7, 3, 0, 10, 3];   // repeats on purpose: the scatter must accumulate

        var rng = new Random(123);
        var wembA = new Tensor(vocab, dim);
        for (int i = 0; i < wembA.Data.Length; i++) wembA.Data[i] = (float)(rng.NextDouble() * 2 - 1);
        var wembB = new Tensor(vocab, dim);
        Array.Copy(wembA.Data, wembB.Data, wembA.Data.Length);

        // Path A — the old formulation: onehot(tokens)·Wemb.
        var oneHot = new Tensor(tokens.Length, vocab);
        for (int i = 0; i < tokens.Length; i++) oneHot.Data[i * vocab + tokens[i]] = 1f;
        var outA = Ag.MatMul(oneHot, wembA);

        // Path B — the new formulation: gather of the token rows.
        var outB = Ag.GatherRows(wembB, tokens);

        Assert.Equal(outA.Data, outB.Data);   // forward bit-identical

        Ag.Mean(outA).Backward();
        Ag.Mean(outB).Backward();
        for (int i = 0; i < wembA.Grad.Length; i++)
            Assert.Equal(wembA.Grad[i], wembB.Grad[i], precision: 7);
    }

    // ── Presets: exact parameter counts ──────────────────────────────────────

    [Fact]
    public void SmallPreset_ParameterCount_MatchesTheFormula()
    {
        var cfg   = DecoderLmConfig.Small(vocab: 16384);
        var model = new DecoderLM(cfg, new Random(1));

        Assert.Equal(cfg.DenseParameterFormula(), model.ParameterCount);
        Assert.Equal(26_767_104L, model.ParameterCount);          // ~30M-class @ 16k vocab
        Assert.Equal(64, cfg.HeadDim);                            // the standard head width
        Assert.Equal(512, cfg.MaxSeq);
    }

    [Fact]
    public void FormulaAgreesWithInstantiation_OnAnArbitraryDenseConfig()
    {
        // Cross-check the closed-form count against a real instance on a config
        // with nothing round about it — guards the formula, cheaply.
        var cfg   = new DecoderLmConfig(Vocab: 301, Dim: 24, NumBlocks: 3, NumHeads: 4, FfDim: 56, MaxSeq: 32);
        var model = new DecoderLM(cfg, new Random(2));

        Assert.Equal(cfg.DenseParameterFormula(), model.ParameterCount);
    }

    [Fact]
    public void BasePreset_FormulaGivesTheGpt2SmallClassCount()
    {
        // Not instantiated (135M params ≈ 1 GB with grads); the formula is
        // verified against real instances by the two tests above.
        var cfg = DecoderLmConfig.Base(vocab: 32768);

        Assert.Equal(135_350_784L, cfg.DenseParameterFormula());  // GPT-2-small class
        Assert.Equal(64, cfg.HeadDim);
        Assert.Equal(1024, cfg.MaxSeq);
    }

    [Fact]
    public void DenseParameterFormula_RefusesNonDenseConfigs()
    {
        var moe = DecoderLmConfig.Small(1000) with { NumExperts = 4 };
        Assert.Throws<InvalidOperationException>(() => moe.DenseParameterFormula());
    }

    // ── The factories wire the tokenizer in ──────────────────────────────────

    [Fact]
    public void CreateSmall_TakesVocabFromTokenizer_AndRuns()
    {
        var llm = PredictiveLlm.CreateSmall(new ByteTokenizer(), maxSeq: 64);

        Assert.Equal(260, llm.Config.Vocab);
        Assert.Equal(384, llm.Config.Dim);
        Assert.Equal(0, llm.Config.NumExperts);   // dense: the showcase features stay off
        Assert.Equal(1, llm.Config.NumLoops);

        var logits = llm.Model.Forward([1, 2, 3]);
        Assert.Equal(3, logits.Rows);
        Assert.Equal(260, logits.Cols);
        Assert.All(logits.Data, v => Assert.True(float.IsFinite(v)));
    }
}
