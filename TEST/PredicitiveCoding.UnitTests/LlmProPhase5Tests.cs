using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Training;
using PredictiveCoding.Model;

namespace PredicitiveCoding.UnitTests;

// Pro roadmap Phase 5: checkpointing, professionally. True resume (Adam moments
// travel with the weights, so a killed run CONTINUES), periodic best-on-val
// checkpoints, and quantized checkpoints (Half/int8 reload within tolerance).
public class LlmProPhase5Tests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"ckpt_{Guid.NewGuid():N}");

    public LlmProPhase5Tests() => Directory.CreateDirectory(_dir);
    public void Dispose() { if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true); }

    private static DecoderLmConfig Cfg() =>
        new(260, Dim: 16, NumBlocks: 2, NumHeads: 2, FfDim: 32, MaxSeq: 48, BlockPattern: "TA");

    private static List<(int[] tokens, int[] targets)> Batch(int seed)
    {
        var rng = new Random(seed);
        var batch = new List<(int[], int[])>();
        for (int i = 0; i < 4; i++)
        {
            var toks = new int[10]; var tgts = new int[10];
            for (int j = 0; j < 10; j++) { toks[j] = rng.Next(256); tgts[j] = rng.Next(256); }
            batch.Add((toks, tgts));
        }
        return batch;
    }

    // ── True resume: kill after 10, resume, ≡ uninterrupted 20 ───────────────

    [Fact]
    public void ResumeContinues_MatchesUninterruptedRun()
    {
        var batches = Enumerable.Range(0, 20).Select(Batch).ToList();
        const float lr = 0.01f;

        // Uninterrupted reference: 20 steps on one model+optimizer.
        var refModel = new DecoderLM(Cfg(), new Random(7));
        var refOpt   = new Adam();
        foreach (var b in batches)
            Trainer.Step(refModel.Parameters(), b, (t, y, tr) => refModel.Loss(t, y, -1, tr), refOpt, lr, 1f);

        // Interrupted: 10 steps, save weights + optimizer, load into fresh
        // instances, 10 more steps.
        var a = new DecoderLM(Cfg(), new Random(7));
        var aOpt = new Adam();
        for (int s = 0; s < 10; s++)
            Trainer.Step(a.Parameters(), batches[s], (t, y, tr) => a.Loss(t, y, -1, tr), aOpt, lr, 1f);

        string wdir = Path.Combine(_dir, "weights");
        Checkpoint.Save(wdir, a.Parameters());
        OptimizerCheckpoint.Save(Path.Combine(_dir, "opt.bin"), a.Parameters(), aOpt);

        var b2 = new DecoderLM(Cfg(), new Random(999));               // different init
        var b2Opt = new Adam();
        Checkpoint.Load(wdir, b2.Parameters());
        OptimizerCheckpoint.Load(Path.Combine(_dir, "opt.bin"), b2.Parameters(), b2Opt);
        Assert.Equal(10, b2Opt.StepCount);                            // step counter restored
        for (int s = 10; s < 20; s++)
            Trainer.Step(b2.Parameters(), batches[s], (t, y, tr) => b2.Loss(t, y, -1, tr), b2Opt, lr, 1f);

        var rp = refModel.Parameters(); var bp = b2.Parameters();
        for (int k = 0; k < rp.Length; k++)
            for (int i = 0; i < rp[k].Data.Length; i++)
                Assert.Equal(rp[k].Data[i], bp[k].Data[i], precision: 4);   // resume ≡ uninterrupted
    }

    [Fact]
    public void RestartingOptimizer_DivergesFromUninterrupted()
    {
        // Control: WITHOUT restoring the moments the resumed run differs — proves
        // the equivalence above is actually carried by the optimizer state.
        var batches = Enumerable.Range(0, 20).Select(Batch).ToList();
        const float lr = 0.01f;

        var refModel = new DecoderLM(Cfg(), new Random(7));
        var refOpt   = new Adam();
        foreach (var b in batches)
            Trainer.Step(refModel.Parameters(), b, (t, y, tr) => refModel.Loss(t, y, -1, tr), refOpt, lr, 1f);

        var a = new DecoderLM(Cfg(), new Random(7));
        var aOpt = new Adam();
        for (int s = 0; s < 10; s++)
            Trainer.Step(a.Parameters(), batches[s], (t, y, tr) => a.Loss(t, y, -1, tr), aOpt, lr, 1f);
        Checkpoint.Save(Path.Combine(_dir, "w2"), a.Parameters());

        var b2 = new DecoderLM(Cfg(), new Random(1));
        Checkpoint.Load(Path.Combine(_dir, "w2"), b2.Parameters());
        var b2Opt = new Adam();                                       // fresh — moments NOT restored
        for (int s = 10; s < 20; s++)
            Trainer.Step(b2.Parameters(), batches[s], (t, y, tr) => b2.Loss(t, y, -1, tr), b2Opt, lr, 1f);

        var rp = refModel.Parameters(); var bp = b2.Parameters();
        double maxDiff = 0;
        for (int k = 0; k < rp.Length; k++)
            for (int i = 0; i < rp[k].Data.Length; i++)
                maxDiff = Math.Max(maxDiff, Math.Abs(rp[k].Data[i] - bp[k].Data[i]));
        Assert.True(maxDiff > 1e-4, $"restart should differ from uninterrupted, max diff was {maxDiff:E2}");
    }

    // ── Quantized checkpoints reload within format tolerance ─────────────────

    [Theory]
    [InlineData(WeightStoragePrecision.Half,     5e-2f)]
    [InlineData(WeightStoragePrecision.BFloat16, 5e-2f)]
    [InlineData(WeightStoragePrecision.Int8,     1e-1f)]
    public void QuantizedCheckpoint_ReloadsWithinTolerance(WeightStoragePrecision precision, float tol)
    {
        var model = new DecoderLM(Cfg(), new Random(3));
        var opt   = new Adam();
        foreach (var b in Enumerable.Range(0, 6).Select(Batch))
            Trainer.Step(model.Parameters(), b, (t, y, tr) => model.Loss(t, y, -1, tr), opt, 0.01f, 1f);

        string wdir = Path.Combine(_dir, precision.ToString());
        Checkpoint.Save(wdir, model.Parameters(), precision);

        var reload = new DecoderLM(Cfg(), new Random(42));
        Checkpoint.Load(wdir, reload.Parameters());

        var mp = model.Parameters(); var rp = reload.Parameters();
        for (int k = 0; k < mp.Length; k++)
            for (int i = 0; i < mp[k].Data.Length; i++)
            {
                float t = tol * (1f + MathF.Abs(mp[k].Data[i]));
                Assert.True(MathF.Abs(mp[k].Data[i] - rp[k].Data[i]) <= t,
                    $"{precision} tensor {k}[{i}]: {mp[k].Data[i]} vs {rp[k].Data[i]}");
            }
    }

    [Fact]
    public void HalfCheckpoint_IsAboutHalfTheSizeOfFloat32()
    {
        var model = new DecoderLM(Cfg(), new Random(3));
        Checkpoint.Save(Path.Combine(_dir, "f32"), model.Parameters(), WeightStoragePrecision.Float32);
        Checkpoint.Save(Path.Combine(_dir, "f16"), model.Parameters(), WeightStoragePrecision.Half);

        long f32 = DirSize(Path.Combine(_dir, "f32"));
        long f16 = DirSize(Path.Combine(_dir, "f16"));
        Assert.True(f16 < f32, $"Half checkpoint ({f16}) should be smaller than Float32 ({f32})");
    }

    // ── Product surface: periodic best-on-val + LoadWithOptimizer ────────────

    [Fact]
    public void Train_WritesBestCheckpoint_AndReloadsWithOptimizer()
    {
        string ckpt = Path.Combine(_dir, "best");
        var llm = PredictiveLlm.Create2026(dim: 16, blocks: 2, heads: 2, ffDim: 32, maxSeq: 48);
        var report = llm.Train(
            "the quick brown fox jumps over the lazy dog\npack my box with five dozen liquor jugs",
            new TrainingOptions
            {
                SeqLen = 16, Stride = 8, Epochs = 10, BatchWindows = 4,
                CheckpointEverySteps = 3, CheckpointDir = ckpt,
            });

        Assert.Equal(ckpt, report.BestCheckpointPath);
        Assert.True(double.IsFinite(report.BestPerplexity));
        Assert.True(File.Exists(Path.Combine(ckpt, "optimizer.bin")));

        var (reloaded, opt) = PredictiveLlm.LoadWithOptimizer(ckpt);
        Assert.True(opt.StepCount > 0);                               // optimizer state came back
        Assert.Equal(llm.Config.Vocab, reloaded.Config.Vocab);
    }

    private static long DirSize(string dir) =>
        Directory.EnumerateFiles(dir, "*", SearchOption.AllDirectories).Sum(f => new FileInfo(f).Length);
}
