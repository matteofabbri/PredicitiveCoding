using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Generation;
using PredictiveCoding.LLM.Model;
using PredictiveCoding.LLM.Tokenization;
using PredictiveCoding.LLM.Training;

namespace PredicitiveCoding.UnitTests;

public class LlmTrainingTests
{
    // ── Learning-rate schedule ─────────────────────────────────────────────────

    [Fact]
    public void LrSchedule_WarmsUpThenDecays()
    {
        var s = new LrSchedule { PeakLr = 1f, MinLr = 0.1f, WarmupSteps = 10, TotalSteps = 100 };
        Assert.True(s.At(0) < s.At(9));               // ramping up
        Assert.Equal(1f, s.At(9), precision: 4);      // peak at end of warmup
        Assert.True(s.At(50) < 1f && s.At(50) > 0.1f); // decaying
        Assert.Equal(0.1f, s.At(100), precision: 4);   // floor after total
    }

    // ── Gradient clipping ───────────────────────────────────────────────────────

    [Fact]
    public void GradientClipping_BoundsGlobalNorm()
    {
        var lm = new TinyTransformerLM(5, 4, 8, new Random(1));
        lm.Loss([1, 2, 3], [2, 3, 4]).Backward();
        var ps = lm.Parameters();

        float pre = GradientClipping.ClipGlobalNorm(ps, 0.5f);
        double sq = 0;
        foreach (var p in ps) foreach (float g in p.Grad) sq += (double)g * g;
        float post = (float)Math.Sqrt(sq);

        Assert.True(pre >= post);
        Assert.True(post <= 0.5f + 1e-3f);
    }

    // ── Adam ────────────────────────────────────────────────────────────────────

    [Fact]
    public void Adam_ReducesLoss()
    {
        var lm = new TinyTransformerLM(6, 8, 16, new Random(2));
        int[] toks = [1, 2, 3, 4], tgts = [2, 3, 4, 5];
        var opt = new Adam(weightDecay: 0.01f);

        float first = lm.Loss(toks, tgts).Data[0];
        for (int it = 0; it < 200; it++)
        {
            lm.Loss(toks, tgts).Backward();
            opt.Step(lm.Parameters(), 0.01f);
        }
        Assert.True(lm.Loss(toks, tgts).Data[0] < first * 0.4f);
    }

    // ── Dropout op ──────────────────────────────────────────────────────────────

    [Fact]
    public void Dropout_MasksInTraining_IdentityInInference()
    {
        var rng = new Random(0);
        var x = new Tensor(1, 1000);
        Array.Fill(x.Data, 1f);

        var train = Ag.Dropout(x, 0.5f, training: true, rng);
        int zeros = train.Data.Count(d => d == 0f);
        Assert.InRange(zeros, 350, 650);                       // ~half dropped
        Assert.Contains(train.Data, d => MathF.Abs(d - 2f) < 1e-5f);  // survivors scaled by 1/(1-p)

        var infer = Ag.Dropout(x, 0.5f, training: false, rng);
        Assert.All(infer.Data, d => Assert.Equal(1f, d));
    }

    [Fact]
    public void Dropout_PassesGradientThrough()
    {
        // p = 0 ⇒ identity; the gradient must match a path without dropout.
        var data = new float[] { 0.2f, -0.5f, 0.3f, 0.1f, 0.4f, -0.2f };
        int[] targets = [0, 2];

        var x1 = Tensor.From(data, 2, 3);
        Ag.CrossEntropy(Ag.Dropout(x1, 0f, training: true, new Random(0)), targets).Backward();

        var x2 = Tensor.From(data, 2, 3);
        Ag.CrossEntropy(x2, targets).Backward();

        for (int i = 0; i < data.Length; i++) Assert.Equal(x2.Grad[i], x1.Grad[i], precision: 5);
    }

    // ── Gradient accumulation (Trainer) ────────────────────────────────────────

    [Fact]
    public void Trainer_AccumulatesAveragedGradients()
    {
        int[] aT = [1, 2, 3], aY = [2, 3, 4];
        int[] bT = [4, 3, 2], bY = [3, 2, 1];

        // Individual gradients from a reference model.
        var refModel = new TinyTransformerLM(5, 4, 8, new Random(9));
        var rp = refModel.Parameters();
        refModel.Loss(aT, aY).Backward();
        var gA = rp.Select(p => (float[])p.Grad.Clone()).ToArray();
        refModel.Loss(bT, bY).Backward();
        var gB = rp.Select(p => (float[])p.Grad.Clone()).ToArray();

        // Same-seed model run through the accumulating Trainer (lr 0 ⇒ weights unchanged).
        var model = new TinyTransformerLM(5, 4, 8, new Random(9));
        Trainer.Step(model, [(aT, aY), (bT, bY)], new Adam(), learningRate: 0f, training: false);
        var ps = model.Parameters();

        for (int k = 0; k < ps.Length; k++)
            for (int i = 0; i < ps[k].Grad.Length; i++)
                Assert.Equal((gA[k][i] + gB[k][i]) / 2f, ps[k].Grad[i], precision: 4);
    }

    [Fact]
    public void Trainer_StepReducesLoss()
    {
        var lm = new TinyTransformerLM(6, 8, 16, new Random(3));
        var batch = new (int[], int[])[] { ([1, 2, 3], [2, 3, 4]), ([4, 5, 1], [5, 1, 2]) };
        var opt = new Adam();

        float first = Trainer.Step(lm, batch, opt, 0.02f, clipNorm: 1f);
        for (int it = 0; it < 60; it++) Trainer.Step(lm, batch, opt, 0.02f, clipNorm: 1f);
        float last = Trainer.Step(lm, batch, opt, 0.02f, clipNorm: 1f);
        Assert.True(last < first);
    }

    // ── Checkpoint ──────────────────────────────────────────────────────────────

    [Fact]
    public void Checkpoint_SaveAndLoad_RestoresWeights()
    {
        int[] t = [1, 2, 3], y = [2, 3, 4];
        var lm = new TinyTransformerLM(5, 4, 8, new Random(6));
        var opt = new Adam();
        for (int it = 0; it < 10; it++) { lm.Loss(t, y).Backward(); opt.Step(lm.Parameters(), 0.05f); }

        float before = lm.Loss(t, y).Data[0];
        string path = Path.Combine(Path.GetTempPath(), $"ckpt_{Guid.NewGuid():N}.bin");
        try
        {
            Checkpoint.Save(path, lm.Parameters());
            foreach (var p in lm.Parameters()) for (int i = 0; i < p.Data.Length; i++) p.Data[i] += 1f;
            Assert.NotEqual(before, lm.Loss(t, y).Data[0], precision: 3);

            Checkpoint.Load(path, lm.Parameters());
            Assert.Equal(before, lm.Loss(t, y).Data[0], precision: 4);
        }
        finally { File.Delete(path); }
    }

    // ── WordPiece ───────────────────────────────────────────────────────────────

    [Fact]
    public void WordPiece_TrainsEncodesDecodes()
    {
        var wp = new WordPieceTokenizer();
        wp.Train("the cat sat on the mat the cat ran after the cat", vocabSize: 60, maxPieceLen: 6);

        var ids = wp.Encode("the cat", addBos: true, addEos: true);
        Assert.Equal(wp.Bos, ids[0]);
        Assert.Equal(wp.Eos, ids[^1]);
        Assert.Equal("the cat", wp.Decode(ids));
        Assert.Contains(wp.Unk, wp.Encode("zzz"));     // 'z' unseen ⇒ UNK
    }

    // ── ALiBi ───────────────────────────────────────────────────────────────────

    [Fact]
    public void Alibi_ProducesFiniteLogits()
    {
        var cfg = new TransformerConfig
        {
            VocabSize = 260, MaxSeqLen = 16, EmbedDim = 32, NumHeads = 4, NumLayers = 2,
            FeedForwardDim = 64, Positional = PositionalKind.Alibi,
        };
        var model  = new GptModel(cfg, new Random(1));
        var logits = model.Step(65, 0, model.CreateCache());
        Assert.Equal(cfg.VocabSize, logits.Length);
        Assert.All(logits.Buffer, f => Assert.True(float.IsFinite(f)));
    }

    // ── Continuous batching ──────────────────────────────────────────────────────

    [Fact]
    public void ContinuousBatcher_MatchesBatchedGreedy()
    {
        var model = new GptModel(
            new TransformerConfig { VocabSize = 260, MaxSeqLen = 32, EmbedDim = 32, NumHeads = 4, NumLayers = 2, FeedForwardDim = 64 },
            new Random(42));
        var tok  = new ByteTokenizer();
        var opts = new SamplingOptions { MaxNewTokens = 6, Greedy = true };

        var prompts = new[] { "a", "bb", "ccc" };
        var encoded = prompts.Select(p => (IReadOnlyList<int>)tok.Encode(p, addBos: true)).ToList();

        var continuous = new ContinuousBatcher(model, tok, maxConcurrency: 2)
            .Run(encoded.Select(e => (e, opts)).ToList());
        var batched = new BatchedGenerator(model, tok).GenerateBatch(encoded, opts);

        Assert.Equal(3, continuous.Count);
        for (int i = 0; i < 3; i++) Assert.Equal(batched[i], continuous[i]);   // greedy ⇒ identical
    }
}
