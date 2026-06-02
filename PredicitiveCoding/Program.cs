using PredictiveCoding.Core;
using PredictiveCoding.Data;
using PredictiveCoding.Logging;
using PredictiveCoding.Model;
using PredictiveCoding.Training;

Console.OutputEncoding = System.Text.Encoding.UTF8;
Console.WriteLine("╔══════════════════════════════════════════════════════╗");
Console.WriteLine("║    Predictive Coding Framework  ·  .NET 10 / C#     ║");
Console.WriteLine("╚══════════════════════════════════════════════════════╝");

// ── 1. Configuration ──────────────────────────────────────────────────────────

var cfg = new TrainingConfig
{
    Network = new NetworkConfig
    {
        // Architecture: input=20, hidden=[64,32], output=4 classes
        LayerSizes = [20, 64, 32, 4],

        // Generative non-linearity: Tanh (bounded, smooth derivative everywhere)
        Activation = ActivationType.Tanh,

        // η_state: large enough to converge inference
        StateLearningRate = 0.05f,

        // η_weight: small for learning stability
        WeightLearningRate = 0.002f,
    },

    // Inference steps to convergence (typical: 10-50)
    InferenceSteps = 30,

    Epochs    = 60,
    BatchSize = 16,
    Seed      = 42,

    CsvOutputPath   = "metrics.csv",
    LogEveryNEpochs = 5
};

cfg.Print();

// ── 2. Synthetic dataset ──────────────────────────────────────────────────────

Console.WriteLine("\n[Generating synthetic dataset...]");

var (trainSet, testSet) = SyntheticDataset.Split(
    numClasses    : cfg.LayerSizes[^1],  // = 4
    inputDim      : cfg.LayerSizes[0],   // = 20
    totalPerClass : 200,
    testFraction  : 0.2f,
    noise         : 0.4f,
    seed          : cfg.Seed);

// ── 3. Network construction ───────────────────────────────────────────────────

var rng = new Random(cfg.Seed);
var net = new Network(cfg.Network, rng);

// ── 4. Training ───────────────────────────────────────────────────────────────

var logger    = new MetricLogger(verbose: true);
using var csv = new CsvLogger(cfg.CsvOutputPath ?? "metrics.csv");

PCTrainer.Train(net, trainSet, testSet, cfg, logger, csv);

// ── 5. Post-training analysis ─────────────────────────────────────────────────

Console.WriteLine("\n[Layer energy analysis...]");
AnalyzeLayerEnergies(net, testSet, cfg, rng);

Console.WriteLine("\n[Sample predictions...]");
ShowSamplePredictions(net, testSet, cfg, n: 10);

Console.WriteLine($"\nMetrics saved to: {cfg.CsvOutputPath}");
Console.WriteLine("Done.");

// ═════════════════════════════════════════════════════════════════════════════
// Analysis helpers
// ═════════════════════════════════════════════════════════════════════════════

static void AnalyzeLayerEnergies(Network net, IDataset testSet, TrainingConfig cfg, Random rng)
{
    int   maxSamples     = Math.Min(50, testSet.TotalSamples);
    var   energiesByLayer = new float[net.Layers.Length - 1];
    int   count           = 0;

    foreach (var batch in testSet.GetBatches(1, rng))
    {
        if (count >= maxSamples) break;
        net.Predict(batch.Inputs[0], cfg.InferenceSteps);
        for (int l = 0; l < net.Layers.Length  - 1; l++)
            energiesByLayer[l] += net.Layers[l].LocalEnergy();
        count++;
    }

    Console.WriteLine($"  Average energy per layer (over {count} test samples):");
    for (int l = 0; l < net.Layers.Length - 1; l++)
        Console.WriteLine($"    ε[{l}]  =  {energiesByLayer[l] / count:F4}");
}

static void ShowSamplePredictions(Network net, IDataset testSet, TrainingConfig cfg, int n)
{
    int shown   = 0;
    int correct = 0;

    Console.WriteLine($"  {"Sample",6} | {"GT",4} | {"Pred",4} | {"OK",3}");
    Console.WriteLine("  " + new string('─', 28));

    foreach (var batch in testSet.GetBatches(1))
    {
        if (shown >= n) break;
        int  gt   = TensorUtils.Argmax(batch.Labels[0]);
        int  pred = net.Predict(batch.Inputs[0], cfg.InferenceSteps);
        bool ok   = gt == pred;
        if (ok) correct++;
        Console.WriteLine($"  {shown + 1,6} | {gt,4} | {pred,4} | {(ok ? "✓" : "✗"),3}");
        shown++;
    }

    Console.WriteLine($"  Accuracy on {n} samples: {(float)correct / n * 100:F1}%");
}
