using System.Diagnostics;
using PredictiveCoding.Core;
using PredictiveCoding.Data;
using PredictiveCoding.Logging;
using PredictiveCoding.Model;
using PredictiveCoding.Training;

namespace PredictiveCoding.Main.Demos;

// ─────────────────────────────────────────────────────────────────────────────
// HardClassificationDemo
//
// Trains a flat Predictive Coding MLP on CIFAR-10 — a benchmark that is
// intrinsically much harder than MNIST:
//
//   • 10 classes of real-world objects (airplane, car, bird, …)
//   • RGB images: 3 × 32 × 32 = 3 072 input features
//   • High intra-class variance, low inter-class separation
//   • No spatial inductive bias (flat MLP must learn everything from pixels)
//
// Expected accuracy for a flat MLP on CIFAR-10:
//   Random baseline : 10 %
//   Linear model    : ~40 %
//   Flat MLP (BP)   : ~50–55 %
//   This PC network : ~35–45 %  (energy-based inference, no convolutions)
//
// Speed note:
//   LazyPersistence = true defers FasterKV writes to FlushStore() at
//   epoch end.  Without this, per-sample writes of the 6 MB weight matrix
//   (3072×512 floats) saturate I/O and make each epoch take minutes.
// ─────────────────────────────────────────────────────────────────────────────

public static class HardClassificationDemo
{
    private const int Seed             = 42;
    private const int TrainSamples     = 10_000;   // 20 % of full CIFAR-10
    private const int EvalSamples      =    400;   // used during training loop
    private const int FinalTestSamples =  2_000;   // per-class + noise eval
    private const int NoiseTestSamples =    400;   // noise robustness subset
    private const int Epochs           =     30;
    private const int InferenceSteps   =     10;
    private const int BatchSize        =     32;
    private const int LogEvery         =      5;

    // Network: 3072 → 512 → 256 → 10
    private static readonly int[] LayerSizes = [3 * 32 * 32, 512, 256, 10];

    public static void Run(string cifarCacheDir = "./cifar-10-data")
    {
        Banner();

        // ── Load dataset ──────────────────────────────────────────────────────

        Console.WriteLine("  Loading CIFAR-10...");
        Console.WriteLine($"  (cache: {Path.GetFullPath(cifarCacheDir)})");
        Console.WriteLine();

        CifarDataset trainFull, testFull;
        try
        {
            (trainFull, testFull) = CifarDataset.Load(
                cifarCacheDir, TrainSamples, FinalTestSamples);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"  ERROR loading CIFAR-10: {ex.Message}");
            Console.WriteLine();
            Console.WriteLine("  To download manually:");
            Console.WriteLine("    https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz");
            Console.WriteLine($"  Extract to: {Path.GetFullPath(cifarCacheDir)}");
            Console.WriteLine("  (the binary version, not Python pickle)");
            return;
        }

        // Small eval split used during training to keep each epoch fast
        var evalSet = testFull.Subset(EvalSamples);

        Console.WriteLine($"  Train : {trainFull.TotalSamples} images");
        Console.WriteLine($"  Test  : {testFull.TotalSamples} images  " +
                          $"(eval during training: {EvalSamples})");
        Console.WriteLine($"  Input : {string.Join("×", CifarDataset.InputShape)} = " +
                          $"{trainFull.InputSize} features");
        Console.WriteLine($"  Random baseline: {100.0 / CifarDataset.ClassNames.Length:F0}%");
        Console.WriteLine();

        // ── Build network ─────────────────────────────────────────────────────

        string netDir = Path.Combine(Path.GetTempPath(), $"pccifarnet_{Guid.NewGuid():N}");
        var netCfg = new NetworkConfig
        {
            Path               = netDir,
            LayerSizes         = LayerSizes,
            Activation         = ActivationType.Tanh,
            StateLearningRate  = 0.05f,
            WeightLearningRate = 0.001f,
            InputShape         = CifarDataset.InputShape,
            LazyPersistence    = true,   // defer FasterKV writes to FlushStore()
        };
        var trainCfg = new TrainingConfig
        {
            Network         = netCfg,
            InferenceSteps  = InferenceSteps,
            Epochs          = Epochs,
            BatchSize       = BatchSize,
            Seed            = Seed,
            LogEveryNEpochs = LogEvery,
        };

        trainCfg.Print();

        var rng = new Random(Seed);
        using var net = new Network(netCfg, rng);

        // ── Section 1: Training curve ─────────────────────────────────────────

        SectionHeader("Section 1", "Training Curve");
        Console.WriteLine("  Free energy F and accuracy per epoch.");
        Console.WriteLine("  Classification: c* = argmin_c F(x, one-hot(c))");
        Console.WriteLine("  LazyPersistence: FasterKV written once per epoch " +
                          "(not per sample).");
        Console.WriteLine();

        var history = TrainWithHistory(net, trainFull, evalSet, trainCfg);
        PrintTrainingCurve(history);

        // ── Section 2: Per-class accuracy ─────────────────────────────────────

        SectionHeader("Section 2", "Per-class Accuracy");
        Console.WriteLine("  Which object categories does the PC network find easy vs. hard?");
        Console.WriteLine($"  (evaluated on {testFull.TotalSamples} test images)");
        Console.WriteLine();

        PrintPerClassAccuracy(net, testFull, trainCfg);

        // ── Section 3: Noise robustness ───────────────────────────────────────

        SectionHeader("Section 3", "Noise Robustness — Pixel Corruption");
        Console.WriteLine("  Additive Gaussian noise N(0, σ²) on each pixel at test time.");
        Console.WriteLine("  Unlike Gaussian blobs, CIFAR-10 shows a real degradation curve.");
        Console.WriteLine();

        PrintNoiseRobustness(net, testFull.Subset(NoiseTestSamples), trainCfg);

        Console.WriteLine();
        Console.WriteLine("  Demo complete.");
        Console.WriteLine(new string('═', 70));

        // Clean up temp dir
        net.Dispose();
        if (Directory.Exists(netDir))
            try { Directory.Delete(netDir, recursive: true); } catch { }
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Training loop with per-epoch flush + history
    // ─────────────────────────────────────────────────────────────────────────

    private record EpochResult(int Epoch, float TrainEnergy, float TestEnergy,
        float TestAccuracy, double ElapsedSec);

    private static List<EpochResult> TrainWithHistory(
        Network net, CifarDataset train, CifarDataset evalSet, TrainingConfig cfg)
    {
        var history  = new List<EpochResult>();
        var trainRng = new Random(cfg.Seed);
        var sw       = Stopwatch.StartNew();

        Console.WriteLine($"  {"Epoch",6}  {"TrainE",9}  {"EvalE",9}  {"EvalAcc",9}  {"t(s)",6}");
        Console.WriteLine("  " + new string('─', 52));

        for (int epoch = 1; epoch <= cfg.Epochs; epoch++)
        {
            sw.Restart();
            var (trainE, _) = PCTrainer.TrainEpoch(net, train, cfg, trainRng);

            // Persist accumulated dirty writes once per epoch
            net.FlushStore();

            bool doEval = epoch % cfg.LogEveryNEpochs == 0
                       || epoch == 1 || epoch == cfg.Epochs;

            if (doEval)
            {
                var (testE, testAcc) = PCTrainer.Evaluate(net, evalSet, cfg);
                double elapsed = sw.Elapsed.TotalSeconds;

                var r = new EpochResult(epoch, trainE, testE, testAcc, elapsed);
                history.Add(r);

                Console.WriteLine($"  {epoch,6}  {trainE,9:F3}  {testE,9:F3}  " +
                                  $"{testAcc * 100,8:F1}%  {elapsed,6:F1}");
            }
        }

        Console.WriteLine();
        return history;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ASCII training curve
    // ─────────────────────────────────────────────────────────────────────────

    private static void PrintTrainingCurve(List<EpochResult> history)
    {
        if (history.Count == 0) return;

        float maxAcc = history.Max(r => r.TestAccuracy);

        Console.WriteLine("  Eval accuracy per logged epoch (each █ ≈ 2.5%):");
        Console.WriteLine();

        foreach (var r in history)
        {
            string bar = AsciiBar(r.TestAccuracy, 1f, width: 40);
            Console.WriteLine($"  Ep {r.Epoch,3}  {r.TestAccuracy * 100,5:F1}%  {bar}");
        }

        Console.WriteLine();
        Console.WriteLine($"  Random baseline : 10.0%");
        Console.WriteLine($"  Best eval acc   : {maxAcc * 100:F1}%");
        Console.WriteLine();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Per-class accuracy
    // ─────────────────────────────────────────────────────────────────────────

    private static void PrintPerClassAccuracy(Network net, CifarDataset test, TrainingConfig cfg)
    {
        int n       = CifarDataset.ClassNames.Length;
        var correct = new int[n];
        var total   = new int[n];

        foreach (var batch in test.GetBatches(1))
        {
            int gt   = TensorUtils.Argmax(batch.Labels[0]);
            int pred = net.Predict(batch.Inputs[0], cfg.InferenceSteps);
            total[gt]++;
            if (pred == gt) correct[gt]++;
        }

        float maxAcc = correct.Zip(total)
            .Max(p => p.Second > 0 ? (float)p.First / p.Second : 0f);

        Console.WriteLine($"  {"Class",-12}  {"Correct",8}  {"Total",7}  {"Acc",6}  Chart");
        Console.WriteLine("  " + new string('─', 65));

        for (int c = 0; c < n; c++)
        {
            float acc  = total[c] > 0 ? (float)correct[c] / total[c] : 0f;
            string bar = AsciiBar(acc, Math.Max(maxAcc, 0.01f), width: 24);
            Console.WriteLine($"  {CifarDataset.ClassNames[c],-12}  {correct[c],8}  " +
                              $"{total[c],7}  {acc * 100,5:F1}%  {bar}");
        }

        Console.WriteLine();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Noise robustness
    // ─────────────────────────────────────────────────────────────────────────

    private static void PrintNoiseRobustness(Network net, CifarDataset subset, TrainingConfig cfg)
    {
        var noiseRng = new Random(Seed + 99);

        float[] sigmas   = [0.00f, 0.05f, 0.10f, 0.20f, 0.35f, 0.50f, 0.75f, 1.00f];
        float   cleanAcc = EvalAccuracy(net, subset, cfg);

        Console.WriteLine($"  {"σ (noise)",10}  {"Accuracy",10}  {"Drop",8}  Chart");
        Console.WriteLine("  " + new string('─', 62));

        foreach (float sigma in sigmas)
        {
            CifarDataset evalSet = sigma < 1e-6f
                ? subset
                : subset.WithGaussianNoise(sigma, new Random(noiseRng.Next()));

            float acc     = EvalAccuracy(net, evalSet, cfg);
            float dropPp  = (cleanAcc - acc) * 100f;
            string dropStr = sigma < 1e-6f ? "baseline" : $"-{dropPp,5:F1} pp";
            string bar     = AsciiBar(acc, 1f, width: 24);
            Console.WriteLine($"  σ={sigma,6:F2}     {acc * 100,8:F1}%  {dropStr,10}  {bar}");
        }

        Console.WriteLine();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Helpers
    // ─────────────────────────────────────────────────────────────────────────

    private static float EvalAccuracy(Network net, CifarDataset set, TrainingConfig cfg)
    {
        int correct = 0, total = 0;
        foreach (var batch in set.GetBatches(1))
        {
            int gt   = TensorUtils.Argmax(batch.Labels[0]);
            int pred = net.Predict(batch.Inputs[0], cfg.InferenceSteps);
            if (pred == gt) correct++;
            total++;
        }
        return total > 0 ? (float)correct / total : 0f;
    }

    private static string AsciiBar(float value, float maxValue, int width)
    {
        if (maxValue <= 0f) return new string('░', width);
        int filled = (int)Math.Round(Math.Clamp(value / maxValue, 0f, 1f) * width);
        return new string('█', filled) + new string('░', width - filled);
    }

    private static void SectionHeader(string id, string title)
    {
        Console.WriteLine(new string('─', 70));
        Console.WriteLine($"  {id}  —  {title}");
        Console.WriteLine(new string('─', 70));
    }

    private static void Banner()
    {
        Console.WriteLine();
        Console.WriteLine(new string('═', 70));
        Console.WriteLine("  CIFAR-10 Hard Classification Demo");
        Console.WriteLine("  Flat Predictive Coding MLP on real-world object recognition");
        Console.WriteLine(new string('═', 70));
        Console.WriteLine();
        Console.WriteLine("  Why CIFAR-10 is intrinsically harder than MNIST or Gaussian blobs:");
        Console.WriteLine("  • 10 object classes with high intra-class variance");
        Console.WriteLine("    (a 'dog' can look completely different from another 'dog')");
        Console.WriteLine("  • RGB images (3×32×32): complex texture and colour structure");
        Console.WriteLine("  • No spatial inductive bias: flat MLP learns from raw pixels");
        Console.WriteLine("  • Overlapping class distributions in pixel space");
        Console.WriteLine("    (a 'cat' and 'dog' share similar low-level features)");
        Console.WriteLine();
    }
}
