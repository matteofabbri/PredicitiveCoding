using PredictiveCoding.Device;
using PredictiveCoding.Device.Training;
using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Data;
using PredictiveCoding.LLM.Tokenization;

namespace PredictiveCoding.Main.Demos;

// ─────────────────────────────────────────────────────────────────────────────
// WebPretrainDemo  —  ROADMAP-Scale D: the training run, launchable.
//
// Pairs with demo 12 (which sediments the shard corpus): trains a 16k BPE on
// the accumulation, builds the Small preset on it, and runs DevicePretrainer on
// the 4070 — gradient-accumulated AdamW, warmup+cosine LR, periodic held-out
// eval, periodic QUALITY checks (bits/byte + a live sample), best-on-val and
// rolling checkpoints, and a JSONL log. Ctrl-C stops gracefully with a final
// checkpoint; relaunching RESUMES (weights + device Adam moments), so the
// multi-day run survives restarts. This is where output quality, not mechanism
// correctness, becomes the metric.
// ─────────────────────────────────────────────────────────────────────────────

public static class WebPretrainDemo
{
    public static void Run()
    {
        string root = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile),
                                   ".predictivecoding");
        string shardDir = Path.Combine(root, "web-corpus");
        string modelDir = Path.Combine(root, "web-model");
        string logPath  = Path.Combine(modelDir, "train.jsonl");

        Console.WriteLine("\n── Web pretraining run (ROADMAP-Scale D) ────────────────────");

        if (!DeviceRuntime.CudaAvailable)
        {
            Console.WriteLine("  No CUDA GPU visible — the real run needs the 4070 (CPU would be ~40× slower).");
            Console.WriteLine("  (On a fresh shell, add the CUDA v12.9 bin to PATH.)");
            return;
        }
        if (!Directory.Exists(shardDir) || Directory.GetFiles(shardDir, "shard-*.txt").Length == 0)
        {
            Console.WriteLine($"  No corpus at {shardDir} — run demo 12 first to sediment one from the web.");
            return;
        }

        var corpus = CorpusSource.FromFolder(shardDir).WithBoilerplateStripped(minWords: 5).Deduplicated();
        long shardBytes = Directory.GetFiles(shardDir, "shard-*.txt").Sum(f => new FileInfo(f).Length);
        Console.WriteLine($"  Corpus: {shardBytes / 1024.0 / 1024.0:F1} MB of shards at {shardDir}");

        Console.Write("  Max steps [0 = until Ctrl-C]: ");
        if (!long.TryParse(Console.ReadLine()?.Trim(), out long maxSteps) || maxSteps < 0) maxSteps = 0;
        Console.Write("  bf16 tensor cores? [Y/n]: ");
        bool bf16 = !string.Equals(Console.ReadLine()?.Trim(), "n", StringComparison.OrdinalIgnoreCase);

        // Fresh run trains the vocab and the model; a resume reloads both.
        PredictiveLlm llm;
        if (Directory.Exists(modelDir) && File.Exists(Path.Combine(modelDir, "config.json")))
        {
            llm = PredictiveLlm.Load(modelDir);
            Console.WriteLine($"  Resuming the model in {modelDir} ({llm.ParameterCount / 1_000_000.0:F1}M params, vocab {llm.Config.Vocab}).");
        }
        else
        {
            Console.WriteLine("  Fresh run — training a 16k BPE on the corpus…");
            var tok = new BpeTokenizer();
            tok.TrainFromCorpus(corpus, targetVocab: 16384,
                onProgress: (done, target) => { if (done % 2048 == 0) Console.Write($"\r    merges {done}/{target}"); });
            Console.WriteLine();
            llm = PredictiveLlm.CreateSmall(tok, maxSeq: 512);
            Console.WriteLine($"  Built Small preset: {llm.ParameterCount / 1_000_000.0:F1}M params, vocab {llm.Config.Vocab}.");
        }

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) =>
        {
            e.Cancel = true;                                   // don't kill the process — let the loop unwind
            if (!cts.IsCancellationRequested)
            {
                Console.WriteLine("\n  Ctrl-C — finishing the current step and writing a final checkpoint…");
                cts.Cancel();
            }
        };

        Console.WriteLine($"  Training ({(bf16 ? "bf16" : "fp32")}). Checkpoints → {modelDir}, log → {logPath}\n");

        var report = DevicePretrainer.Pretrain(llm, corpus, new DevicePretrainOptions
        {
            Epochs = 1000,                                     // MaxSteps / Ctrl-C is the real limiter
            SeqLen = 512, Stride = 512,
            MicroBatchWindows = bf16 ? 8 : 4,
            PeakLr = 3e-4f, MinLr = 3e-5f, WarmupSteps = 200,
            MaxSteps = maxSteps,
            EvalEverySteps = 100, EvalWindows = 64,
            QualityEverySteps = 500, SamplePrompt = "The ", SampleTokens = 48,
            CheckpointDir = modelDir, CheckpointEverySteps = 500, KeepBestCheckpoint = true,
            LogPath = logPath, MixedPrecision = bf16,
            OnProgress = PrintProgress,
        }, cts.Token);

        Console.WriteLine($"\n  Done: {report.Steps} steps, {report.Tokens / 1_000_000.0:F1}M tokens, {report.Elapsed:hh\\:mm\\:ss}.");
        Console.WriteLine($"  Final eval loss {report.FinalEvalLoss:F3}, best {report.BestEvalLoss:F3}" +
                          (report.FinalBitsPerByte is double b ? $", {b:F3} bits/byte." : "."));
        Console.WriteLine($"  Best checkpoint → {report.BestCheckpointDir}");
        Console.WriteLine("  Serve it: PredictiveLlm.Load(dir).CreateServer(), or Generate(prompt).");
    }

    private static void PrintProgress(DevicePretrainProgress p)
    {
        string line = $"  step {p.Step,6}  lr {p.Lr:E1}  loss {p.TrainLoss:F3}";
        if (p.EvalLoss is float e) line += $"  eval {e:F3}";
        if (p.BitsPerByte is double b) line += $"  b/byte {b:F3}";
        line += $"  {p.TokensPerSecond,6:F0} tok/s";
        if (p.EtaSeconds is double eta) line += $"  eta {TimeSpan.FromSeconds(eta):hh\\:mm\\:ss}";
        Console.WriteLine(line);
        if (p.Sample is not null) Console.WriteLine($"      ↪ {p.Sample}");
    }
}
