using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Data;
using PredictiveCoding.LLM.Eval;
using PredictiveCoding.LLM.Model;
using PredictiveCoding.LLM.Tokenization;
using PredictiveCoding.LLM.Training;

namespace PredicitiveCoding.UnitTests;

// Phase 0 of PredictiveCoding.LLM/ROADMAP.md: the yardstick — perplexity
// harness + activation observatory. Numbers, not vibes.
public class LlmEvalTests
{
    private static DecoderLM TinyLm(int vocab = 31, int dim = 16, int blocks = 2, int heads = 2, int seed = 1)
        => new(new DecoderLmConfig(vocab, dim, blocks, heads, FfDim: 2 * dim, MaxSeq: 16), new Random(seed));

    // ── Perplexity ───────────────────────────────────────────────────────────

    [Fact]
    public void Perplexity_AllZeroWeights_EqualsVocabSize()
    {
        // Zeroed parameters ⇒ all-zero logits ⇒ exactly uniform prediction:
        // perplexity must equal the vocabulary size.
        var lm = TinyLm(vocab: 11, dim: 8, blocks: 1);
        foreach (var p in lm.Parameters()) Array.Clear(p.Data);

        var r = PerplexityEvaluator.Evaluate(lm, [([1, 2, 3, 4], [2, 3, 4, 5])]);

        Assert.Equal(11.0, r.Perplexity, 3);
        Assert.Equal(4, r.Tokens);
        Assert.Equal(1, r.Windows);
    }

    [Fact]
    public void Perplexity_IsDeterministic_AndSkipsIgnoredTargets()
    {
        var lm = TinyLm();
        var windows = new (int[] input, int[] target)[] { ([1, 2, 3, 4], [2, -1, 4, -1]) };

        var a = PerplexityEvaluator.Evaluate(lm, windows, ignoreIndex: -1);
        var b = PerplexityEvaluator.Evaluate(lm, windows, ignoreIndex: -1);

        Assert.Equal(a, b);                            // no dropout at eval ⇒ bit-stable
        Assert.Equal(2, a.Tokens);                     // two targets masked out
        Assert.True(double.IsFinite(a.Perplexity) && a.Perplexity > 1.0);
    }

    [Fact]
    public void Perplexity_TrainingOnTheCorpus_ReducesIt()
    {
        // The full Phase 0 pipeline: EvalCorpus → StreamingTextDataset →
        // held-out split → perplexity before/after a short training run.
        var tok  = new ByteTokenizer();
        var lm   = new DecoderLM(new DecoderLmConfig(tok.VocabSize, Dim: 16, NumBlocks: 1, NumHeads: 2, FfDim: 32, MaxSeq: 12), new Random(7));
        var data = new StreamingTextDataset(EvalCorpus.TinyLines, tok, seqLen: 12, stride: 6);

        var evalSet  = PerplexityEvaluator.HeldOut(data.Windows(), every: 5).Take(6).ToList();
        var trainSet = PerplexityEvaluator.TrainingSplit(data.Windows(), every: 5).Take(10).ToList();
        var opt      = new Adam();

        var before = PerplexityEvaluator.Evaluate(lm, evalSet);
        for (int step = 0; step < 30; step++)
            Trainer.Step(lm.Parameters(), trainSet, (t, y, tr) => lm.Loss(t, y, -1, tr), opt, learningRate: 0.01f, clipNorm: 1f);
        var after = PerplexityEvaluator.Evaluate(lm, evalSet);

        Assert.True(after.Perplexity < before.Perplexity,
            $"perplexity did not improve: {before.Perplexity:F1} → {after.Perplexity:F1}");
    }

    [Fact]
    public void HeldOutAndTrainingSplit_PartitionTheStream()
    {
        var windows = Enumerable.Range(0, 10).Select(i => (new[] { i }, new[] { i })).ToList();

        var held  = PerplexityEvaluator.HeldOut(windows, every: 3).ToList();
        var train = PerplexityEvaluator.TrainingSplit(windows, every: 3).ToList();

        Assert.Equal([0, 3, 6, 9], held.Select(w => w.input[0]));
        Assert.Equal([1, 2, 4, 5, 7, 8], train.Select(w => w.input[0]));
        Assert.Equal(10, held.Count + train.Count);
    }

    // ── Activation observatory ───────────────────────────────────────────────

    [Fact]
    public void ActivationProbe_DecoderLm_RecordsBlockStatsAndSinkShares()
    {
        var lm    = TinyLm(blocks: 2, heads: 2);
        var probe = new ActivationProbe();
        int[] tokens = [1, 5, 9, 2];

        lm.Forward(tokens, training: false, probe);

        Assert.Equal(2, probe.Blocks.Count);                    // one entry per block
        Assert.All(probe.Blocks, b =>
        {
            Assert.True(float.IsFinite(b.MeanAbs) && float.IsFinite(b.MaxAbs));
            Assert.True(b.MaxAbs >= b.MeanAbs && b.MeanAbs > 0f);
            Assert.True(b.SpikeRatio >= 1f);
        });

        Assert.Equal(2 * 2, probe.Sinks.Count);                 // blocks × heads
        Assert.All(probe.Sinks, s =>
        {
            Assert.InRange(s.SinkShare, 0f, 1f);
            // Row 0 can only attend to position 0, so the share is ≥ 1/seq.
            Assert.True(s.SinkShare >= 1f / tokens.Length - 1e-4f);
        });

        probe.Reset();
        Assert.Empty(probe.Blocks);
        Assert.Empty(probe.Sinks);
    }

    [Fact]
    public void ActivationProbe_GptModelStep_RecordsOneEntryPerLayer()
    {
        var cfg = new TransformerConfig
        {
            VocabSize = 260, MaxSeqLen = 8, EmbedDim = 16, NumHeads = 2,
            NumLayers = 3, FeedForwardDim = 32,
        };
        var model = new GptModel(cfg, new Random(4));
        var probe = new ActivationProbe();

        model.Step(65, 0, model.CreateCache(), probe);

        Assert.Equal(3, probe.Blocks.Count);
        Assert.All(probe.Blocks, b => Assert.True(float.IsFinite(b.MeanAbs) && b.MaxAbs >= b.MeanAbs));
    }

    [Fact]
    public void ActivationProbe_Null_LeavesForwardIdentical()
    {
        // Same seed, with and without probe: the logits must be bit-identical —
        // the probe only observes.
        var a = TinyLm(seed: 11).Forward([1, 2, 3], training: false);
        var b = TinyLm(seed: 11).Forward([1, 2, 3], training: false, new ActivationProbe());

        Assert.Equal(a.Data.Length, b.Data.Length);
        for (int i = 0; i < a.Data.Length; i++)
            Assert.Equal(a.Data[i], b.Data[i]);
    }
}
