using System.Diagnostics;
using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Data;
using PredictiveCoding.LLM.Eval;

namespace PredictiveCoding.Device.Training;

/// <summary>Bootstrap-run knobs. Defaults follow the small-LM pretraining playbook.</summary>
public sealed record DevicePretrainOptions
{
    public int Epochs { get; init; } = 1;
    public int SeqLen { get; init; } = 512;
    public int Stride { get; init; } = 512;

    /// <summary>Windows whose gradients accumulate into one AdamW step.</summary>
    public int MicroBatchWindows { get; init; } = 4;

    public float PeakLr { get; init; } = 3e-4f;
    public float MinLr { get; init; } = 3e-5f;
    public int WarmupSteps { get; init; } = 100;

    /// <summary>Total optimizer steps for the cosine decay; 0 = unknown → constant PeakLr after warmup.</summary>
    public long TotalSteps { get; init; }

    /// <summary>AdamW decoupled weight decay (0 for plain Adam).</summary>
    public float WeightDecay { get; init; } = 0.01f;

    /// <summary>1-in-N windows are held out by content hash (Phase-4: stable across runs).</summary>
    public int HeldOutEvery { get; init; } = 10;

    /// <summary>How many held-out windows the periodic eval averages over.</summary>
    public int EvalWindows { get; init; } = 64;

    public int EvalEverySteps { get; init; } = 200;

    /// <summary>Folder for periodic checkpoints (model + tokenizer + device Adam state); null = none.</summary>
    public string? CheckpointDir { get; init; }

    public int CheckpointEverySteps { get; init; } = 1000;

    /// <summary>bf16 tensor-core GEMMs (A2) instead of fp32.</summary>
    public bool MixedPrecision { get; init; }

    public Action<DevicePretrainProgress>? OnProgress { get; init; }
}

/// <summary>One progress beat of the bootstrap run.</summary>
public sealed record DevicePretrainProgress(int Epoch, long Step, float Lr, float TrainLoss,
                                            float? EvalLoss, long Tokens, double TokensPerSecond);

/// <summary>What the run did.</summary>
public sealed record DevicePretrainReport(long Steps, long Tokens, float FinalTrainLoss,
                                          float FinalEvalLoss, TimeSpan Elapsed, string? CheckpointDir);

// ─────────────────────────────────────────────────────────────────────────────
// DevicePretrainer  —  ROADMAP-Scale C3: the bootstrap, wired end to end.
//
// Streams a CorpusSource (typically the crawler's shard folder plus hygiene
// decorators) through the model's tokenizer into fixed windows, splits them by
// content hash (the SAME windows stay in eval across runs and re-crawls —
// Phase-4's comparability guarantee), and trains the DeviceDecoderTrainer on
// the 4070: gradient-accumulated AdamW steps, linear-warmup + cosine LR,
// periodic held-out eval (forward only), periodic checkpoints with the device
// Adam state saved next to the model so a resumed run CONTINUES. Multi-epoch
// by construction — the "learn to read" pass a once-through stream can't give.
// ─────────────────────────────────────────────────────────────────────────────

public static class DevicePretrainer
{
    public static DevicePretrainReport Pretrain(PredictiveLlm llm, CorpusSource corpus,
                                                DevicePretrainOptions? options = null,
                                                CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(llm);
        ArgumentNullException.ThrowIfNull(corpus);
        var o = options ?? new DevicePretrainOptions();
        if (o.SeqLen > llm.Config.MaxSeq)
            throw new ArgumentOutOfRangeException(nameof(options), $"SeqLen {o.SeqLen} exceeds the model's MaxSeq {llm.Config.MaxSeq}.");

        using var trainer = new DeviceDecoderTrainer(llm.Model, weightDecay: o.WeightDecay,
                                                     mixedPrecision: o.MixedPrecision);

        // Resume: if the checkpoint dir already holds a device Adam state, keep going.
        string? adamPath = o.CheckpointDir is null ? null : Path.Combine(o.CheckpointDir, "adam-device.bin");
        if (adamPath is not null && File.Exists(adamPath)) trainer.LoadState(adamPath);

        // Held-out windows: hashed split, capped — Take stops the enumeration early.
        var heldOut = PerplexityEvaluator.HeldOutHashed(
                          new StreamingTextDataset(corpus.Lines(), llm.Tokenizer, o.SeqLen, o.Stride).Windows(),
                          o.HeldOutEvery)
                      .Take(o.EvalWindows).ToList();

        var sw = Stopwatch.StartNew();
        long step = 0, tokens = 0;
        float trainLoss = float.NaN, evalLoss = float.NaN;
        var batch = new List<(int[] tokens, int[] targets)>(o.MicroBatchWindows);

        for (int epoch = 0; epoch < o.Epochs && !cancellationToken.IsCancellationRequested; epoch++)
        {
            var windows = PerplexityEvaluator.TrainingSplitHashed(
                new StreamingTextDataset(corpus.Lines(), llm.Tokenizer, o.SeqLen, o.Stride).Windows(),
                o.HeldOutEvery);

            foreach (var w in windows)
            {
                if (cancellationToken.IsCancellationRequested) break;
                batch.Add(w);
                if (batch.Count < o.MicroBatchWindows) continue;

                step++;
                float lr = LrAt(step, o);
                trainLoss = trainer.Step(batch, lr);
                tokens += batch.Sum(b => b.tokens.Length);
                batch.Clear();

                bool evalDue = heldOut.Count > 0 && o.EvalEverySteps > 0 && step % o.EvalEverySteps == 0;
                if (evalDue) evalLoss = Evaluate(trainer, heldOut);

                if (o.CheckpointDir is not null && o.CheckpointEverySteps > 0 && step % o.CheckpointEverySteps == 0)
                    Checkpoint(trainer, llm, o.CheckpointDir, adamPath!);

                o.OnProgress?.Invoke(new DevicePretrainProgress(
                    epoch, step, lr, trainLoss, evalDue ? evalLoss : null,
                    tokens, tokens / Math.Max(1e-9, sw.Elapsed.TotalSeconds)));
            }
        }

        // Trailing partial micro-batch: train on it too (nothing is thrown away).
        if (batch.Count > 0 && !cancellationToken.IsCancellationRequested)
        {
            step++;
            trainLoss = trainer.Step(batch, LrAt(step, o));
            tokens += batch.Sum(b => b.tokens.Length);
        }

        if (heldOut.Count > 0) evalLoss = Evaluate(trainer, heldOut);
        if (o.CheckpointDir is not null) Checkpoint(trainer, llm, o.CheckpointDir, adamPath!);
        else trainer.ExportTo(llm.Model);

        return new DevicePretrainReport(step, tokens, trainLoss, evalLoss, sw.Elapsed, o.CheckpointDir);
    }

    private static float Evaluate(DeviceDecoderTrainer trainer, List<(int[] tokens, int[] targets)> heldOut)
    {
        float sum = 0f;
        foreach (var (tok, tgt) in heldOut) sum += trainer.EvaluateLoss(tok, tgt);
        return sum / heldOut.Count;
    }

    private static void Checkpoint(DeviceDecoderTrainer trainer, PredictiveLlm llm, string dir, string adamPath)
    {
        trainer.ExportTo(llm.Model);
        llm.Save(dir);
        trainer.SaveState(adamPath);
    }

    // Linear warmup to PeakLr, then cosine to MinLr over TotalSteps (constant
    // PeakLr when the horizon is unknown).
    private static float LrAt(long step, DevicePretrainOptions o)
    {
        if (o.WarmupSteps > 0 && step <= o.WarmupSteps)
            return o.PeakLr * step / o.WarmupSteps;
        if (o.TotalSteps <= 0 || step >= o.TotalSteps) return o.TotalSteps <= 0 ? o.PeakLr : o.MinLr;
        double t = (step - o.WarmupSteps) / (double)Math.Max(1, o.TotalSteps - o.WarmupSteps);
        return o.MinLr + (o.PeakLr - o.MinLr) * 0.5f * (float)(1 + Math.Cos(Math.PI * t));
    }
}
