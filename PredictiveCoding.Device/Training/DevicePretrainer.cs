using System.Diagnostics;
using System.Globalization;
using System.Text;
using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Data;
using PredictiveCoding.LLM.Eval;

namespace PredictiveCoding.Device.Training;

/// <summary>Pretraining-run knobs. Defaults follow the small-LM pretraining playbook.</summary>
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

    /// <summary>Total optimizer steps for the cosine decay; 0 = fall back to <see cref="MaxSteps"/>, else constant PeakLr.</summary>
    public long TotalSteps { get; init; }

    /// <summary>Hard stop after this many optimizer steps (0 = run the full epochs). Also the ETA/cosine horizon.</summary>
    public long MaxSteps { get; init; }

    /// <summary>AdamW decoupled weight decay (0 for plain Adam).</summary>
    public float WeightDecay { get; init; } = 0.01f;

    /// <summary>1-in-N windows are held out by content hash (Phase-4: stable across runs).</summary>
    public int HeldOutEvery { get; init; } = 10;

    /// <summary>How many held-out windows the periodic eval averages over.</summary>
    public int EvalWindows { get; init; } = 64;

    public int EvalEverySteps { get; init; } = 200;

    // ── Quality (D1): the run's real metric — bits/byte + a sample ───────────

    /// <summary>Steps between quality checks (bits/byte + a generated sample); 0 = off.</summary>
    public int QualityEverySteps { get; init; }

    public string SamplePrompt { get; init; } = "The";
    public int SampleTokens { get; init; } = 48;

    // ── Checkpointing (D1) ────────────────────────────────────────────────────

    /// <summary>Folder for periodic checkpoints (model + tokenizer + device Adam state); null = none.</summary>
    public string? CheckpointDir { get; init; }

    public int CheckpointEverySteps { get; init; } = 1000;

    /// <summary>Also keep the lowest-eval-loss checkpoint under <c>CheckpointDir/best</c>.</summary>
    public bool KeepBestCheckpoint { get; init; } = true;

    // ── Monitoring (D2) ───────────────────────────────────────────────────────

    /// <summary>Append a JSONL row per progress beat here (survives + resumes); null = off.</summary>
    public string? LogPath { get; init; }

    /// <summary>bf16 tensor-core GEMMs (A2) instead of fp32.</summary>
    public bool MixedPrecision { get; init; }

    public Action<DevicePretrainProgress>? OnProgress { get; init; }
}

/// <summary>One progress beat of the run (nullable fields fire only on their cadence).</summary>
public sealed record DevicePretrainProgress(int Epoch, long Step, float Lr, float TrainLoss,
                                            float? EvalLoss, double? BitsPerByte, string? Sample,
                                            float BestEvalLoss, long Tokens, double TokensPerSecond,
                                            double WallSeconds, double? EtaSeconds);

/// <summary>What the run did.</summary>
public sealed record DevicePretrainReport(long Steps, long Tokens, float FinalTrainLoss,
                                          float FinalEvalLoss, float BestEvalLoss, double? FinalBitsPerByte,
                                          TimeSpan Elapsed, string? CheckpointDir, string? BestCheckpointDir);

// ─────────────────────────────────────────────────────────────────────────────
// DevicePretrainer  —  ROADMAP-Scale C3 (the bootstrap) + D1/D2 (the real run).
//
// Streams a CorpusSource (typically the crawler's shard folder plus hygiene
// decorators) through the model's tokenizer into fixed windows, splits them by
// content hash (the SAME windows stay in eval across runs and re-crawls —
// Phase-4's comparability guarantee), and trains the DeviceDecoderTrainer on
// the 4070: gradient-accumulated AdamW steps, linear-warmup + cosine LR.
//
// Phase D turns output QUALITY into the metric, not mechanism correctness:
//   • periodic held-out eval (device forward-only, cheap),
//   • periodic QUALITY checks — bits/byte (tokenizer-independent) + a generated
//     sample, so you can literally watch it learn to read,
//   • best-on-validation checkpoint kept apart from the rolling one,
//   • a JSONL training log that survives and appends across resumes,
//   • true resume: the device Adam moments reload, the run CONTINUES.
// The multi-day run itself is execution; everything it needs is here.
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

        string? checkpointDir = o.CheckpointDir;
        string? adamPath = checkpointDir is null ? null : Path.Combine(checkpointDir, "adam-device.bin");
        string? bestDir = checkpointDir is null || !o.KeepBestCheckpoint ? null : Path.Combine(checkpointDir, "best");
        string? bestAdam = bestDir is null ? null : Path.Combine(bestDir, "adam-device.bin");

        // Resume: reload the device Adam state if it is already sitting there.
        if (adamPath is not null && File.Exists(adamPath)) trainer.LoadState(adamPath);

        // Held-out windows (hashed split, capped) + a text probe for bits/byte,
        // decoded from the first few of them so it stays stable across runs.
        var heldOut = PerplexityEvaluator.HeldOutHashed(
                          new StreamingTextDataset(corpus.Lines(), llm.Tokenizer, o.SeqLen, o.Stride).Windows(),
                          o.HeldOutEvery)
                      .Take(o.EvalWindows).ToList();
        string probeText = BuildProbe(llm, heldOut);

        // The cosine horizon: explicit TotalSteps, else MaxSteps, else open.
        long horizon = o.TotalSteps > 0 ? o.TotalSteps : o.MaxSteps;

        var sw = Stopwatch.StartNew();
        long step = 0, tokens = 0;
        float trainLoss = float.NaN, evalLoss = float.NaN, bestEval = float.PositiveInfinity;
        double? lastBits = null;
        var batch = new List<(int[] tokens, int[] targets)>(o.MicroBatchWindows);
        bool stop = false;

        for (int epoch = 0; epoch < o.Epochs && !stop && !cancellationToken.IsCancellationRequested; epoch++)
        {
            var windows = PerplexityEvaluator.TrainingSplitHashed(
                new StreamingTextDataset(corpus.Lines(), llm.Tokenizer, o.SeqLen, o.Stride).Windows(),
                o.HeldOutEvery);

            foreach (var w in windows)
            {
                if (stop || cancellationToken.IsCancellationRequested) break;
                batch.Add(w);
                if (batch.Count < o.MicroBatchWindows) continue;

                step++;
                float lr = LrAt(step, horizon, o);
                trainLoss = trainer.Step(batch, lr);
                tokens += batch.Sum(b => b.tokens.Length);
                batch.Clear();

                bool evalDue    = heldOut.Count > 0 && o.EvalEverySteps > 0 && step % o.EvalEverySteps == 0;
                bool qualityDue = o.QualityEverySteps > 0 && step % o.QualityEverySteps == 0;
                bool ckptDue    = checkpointDir is not null && o.CheckpointEverySteps > 0 && step % o.CheckpointEverySteps == 0;

                if (evalDue) evalLoss = Evaluate(trainer, heldOut);
                bool newBest = evalDue && bestDir is not null && evalLoss < bestEval;

                // One ExportTo covers whatever needs the CPU-side weights this step.
                double? bits = null; string? sample = null;
                if (qualityDue || ckptDue || newBest)
                {
                    trainer.ExportTo(llm.Model);
                    if (qualityDue)
                    {
                        bits = llm.EvaluateBitsPerByte(probeText).BitsPerByte;
                        lastBits = bits;
                        sample = Sample(llm, o);
                    }
                    if (newBest)
                    {
                        bestEval = evalLoss;
                        llm.Save(bestDir!);
                        trainer.SaveState(bestAdam!);
                    }
                    if (ckptDue)
                    {
                        llm.Save(checkpointDir!);
                        trainer.SaveState(adamPath!);
                    }
                }

                double wall = sw.Elapsed.TotalSeconds;
                double tokPerSec = tokens / Math.Max(1e-9, wall);
                double? eta = horizon > 0 && step < horizon ? (horizon - step) * (wall / step) : null;

                var progress = new DevicePretrainProgress(epoch, step, lr, trainLoss,
                    evalDue ? evalLoss : null, bits, sample,
                    float.IsPositiveInfinity(bestEval) ? float.NaN : bestEval,
                    tokens, tokPerSec, wall, eta);
                if (o.LogPath is not null) AppendLog(o.LogPath, progress);
                o.OnProgress?.Invoke(progress);

                if (o.MaxSteps > 0 && step >= o.MaxSteps) stop = true;
            }
        }

        // Trailing partial micro-batch: train on it too (nothing is thrown away).
        if (batch.Count > 0 && !stop && !cancellationToken.IsCancellationRequested)
        {
            step++;
            trainLoss = trainer.Step(batch, LrAt(step, horizon, o));
            tokens += batch.Sum(b => b.tokens.Length);
        }

        if (heldOut.Count > 0) evalLoss = Evaluate(trainer, heldOut);
        trainer.ExportTo(llm.Model);
        if (checkpointDir is not null)
        {
            llm.Save(checkpointDir);
            trainer.SaveState(adamPath!);
        }

        return new DevicePretrainReport(step, tokens, trainLoss, evalLoss,
            float.IsPositiveInfinity(bestEval) ? evalLoss : bestEval, lastBits,
            sw.Elapsed, checkpointDir, bestDir);
    }

    private static float Evaluate(DeviceDecoderTrainer trainer, List<(int[] tokens, int[] targets)> heldOut)
    {
        float sum = 0f;
        foreach (var (tok, tgt) in heldOut) sum += trainer.EvaluateLoss(tok, tgt);
        return sum / heldOut.Count;
    }

    private static string Sample(PredictiveLlm llm, DevicePretrainOptions o)
    {
        try { return llm.Generate(o.SamplePrompt, o.SampleTokens).ReplaceLineEndings(" ").Trim(); }
        catch { return "(generation failed)"; }
    }

    // A short held-out text for bits/byte, decoded from the first held-out
    // windows (empty when there is no held-out slice — bits/byte then stays off).
    private static string BuildProbe(PredictiveLlm llm, List<(int[] input, int[] target)> heldOut)
    {
        if (heldOut.Count == 0) return string.Empty;
        var sb = new StringBuilder();
        foreach (var (_, target) in heldOut.Take(4))
            sb.AppendLine(llm.Tokenizer.Decode(target));
        return sb.ToString();
    }

    private static void AppendLog(string path, DevicePretrainProgress p)
    {
        var dir = Path.GetDirectoryName(path);
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        var c = CultureInfo.InvariantCulture;
        var sb = new StringBuilder("{");
        sb.Append("\"step\":").Append(p.Step);
        sb.Append(",\"epoch\":").Append(p.Epoch);
        sb.Append(",\"lr\":").Append(p.Lr.ToString("R", c));
        sb.Append(",\"trainLoss\":").Append(p.TrainLoss.ToString("R", c));
        if (p.EvalLoss is float e) sb.Append(",\"evalLoss\":").Append(e.ToString("R", c));
        if (p.BitsPerByte is double b) sb.Append(",\"bitsPerByte\":").Append(b.ToString("R", c));
        sb.Append(",\"tokens\":").Append(p.Tokens);
        sb.Append(",\"tokPerSec\":").Append(p.TokensPerSecond.ToString("F1", c));
        sb.Append(",\"wallSeconds\":").Append(p.WallSeconds.ToString("F1", c));
        sb.Append('}');
        File.AppendAllText(path, sb.ToString() + Environment.NewLine);
    }

    // Linear warmup to PeakLr, then cosine to MinLr over the horizon (constant
    // PeakLr when the horizon is unknown).
    private static float LrAt(long step, long horizon, DevicePretrainOptions o)
    {
        if (o.WarmupSteps > 0 && step <= o.WarmupSteps)
            return o.PeakLr * step / o.WarmupSteps;
        if (horizon <= 0 || step >= horizon) return horizon <= 0 ? o.PeakLr : o.MinLr;
        double t = (step - o.WarmupSteps) / (double)Math.Max(1, horizon - o.WarmupSteps);
        return o.MinLr + (o.PeakLr - o.MinLr) * 0.5f * (float)(1 + Math.Cos(Math.PI * t));
    }
}
