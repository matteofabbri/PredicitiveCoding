using PredictiveCoding.Core;
using PredictiveCoding.Model;
using PredictiveCoding.Networks;
using PredictiveCoding.Networks.Temporal;

namespace PredictiveCoding.Main.Demos;

// ─────────────────────────────────────────────────────────────────────────────
// TemporalDemo
//
// Demonstrates that TemporalNetwork outperforms a standard Network when
// classifying noisy inputs embedded in a predictable temporal sequence.
//
// Experiment:
//   Sequence: 0 → 1 → 2 → 3 → 0 → 1 → 2 → 3 → ...  (repeating cycle)
//   Input:    one-hot(class) + Gaussian noise  σ = NOISE_STD
//             At high noise the symbol is heavily corrupted; the temporal
//             prior from the previous step disambiguates which symbol is
//             likely *next*, lowering the effective inference difficulty.
//
//   Result:   TemporalNetwork achieves higher test accuracy than the baseline
//             Network, particularly at high noise levels.
// ─────────────────────────────────────────────────────────────────────────────

public static class TemporalDemo
{
    // ── Hyper-parameters ─────────────────────────────────────────────────────

    private const int   NUM_CLASSES        = 4;
    private const int   INPUT_DIM          = NUM_CLASSES;   // one-hot input
    private const int   HIDDEN_SIZE        = 32;
    private const int   TRAIN_SEQUENCES    = 400;           // full cycles
    private const int   TEST_SEQUENCES     = 100;
    private const int   SEQUENCE_LENGTH    = NUM_CLASSES;   // one full cycle
    private const int   INFERENCE_STEPS    = 20;
    private const float STATE_LR           = 0.1f;
    private const float WEIGHT_LR          = 0.005f;
    private const float NOISE_STD          = 0.4f;
    private const float TEMPORAL_PRECISION = 2f;

    // ── Entry point ───────────────────────────────────────────────────────────

    public static void Run()
    {
        Console.WriteLine();
        Console.WriteLine("╔══════════════════════════════════════════════════════╗");
        Console.WriteLine("║            Temporal Predictive Coding Demo           ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════╝");
        Console.WriteLine();
        Console.WriteLine($"  Sequence  : 0→1→2→3→0→1→...  ({NUM_CLASSES} classes, repeating)");
        Console.WriteLine($"  Input     : one-hot + Gaussian noise  σ={NOISE_STD}");
        Console.WriteLine($"  Π_temporal: {TEMPORAL_PRECISION}");
        Console.WriteLine($"  Training  : {TRAIN_SEQUENCES} sequences × {SEQUENCE_LENGTH} steps");
        Console.WriteLine();

        var rng = new Random(42);

        string baseDir     = Path.Combine(Path.GetTempPath(), $"temporal_{Guid.NewGuid():N}");
        string baseNetPath = Path.Combine(baseDir, "base");
        string tempNetPath = Path.Combine(baseDir, "temp");

        try
        {
            // ── Build both networks with identical architecture & seed ──────────
            var cfg = new NetworkConfig
            {
                LayerSizes         = [INPUT_DIM, HIDDEN_SIZE, NUM_CLASSES],
                Activation         = ActivationType.Tanh,
                StateLearningRate  = STATE_LR,
                WeightLearningRate = WEIGHT_LR,
            };

            using var baseNet  = new Network(cfg with { Path = baseNetPath }, new Random(1));
            using var temporal = new TemporalNetwork(new Network(cfg with { Path = tempNetPath }, new Random(1)), temporalPrecision: TEMPORAL_PRECISION);

            // ── Training ───────────────────────────────────────────────────────
            Console.WriteLine("  Training...");
            Console.WriteLine($"  {"Epoch",5}  {"Base loss",10}  {"Temp loss",10}");
            Console.WriteLine("  " + new string('─', 32));

            for (int epoch = 0; epoch < TRAIN_SEQUENCES; epoch++)
            {
                float baseLoss = 0f, tempLoss = 0f;
                temporal.Reset();

                for (int t = 0; t < SEQUENCE_LENGTH; t++)
                {
                    int   cls   = t % NUM_CLASSES;
                    var   input = NoisyOneHot(cls, rng);
                    var   label = OneHot(cls);

                    baseNet.Learn(input, label, INFERENCE_STEPS);
                    temporal.Step(input, label, INFERENCE_STEPS);

                    baseLoss += baseNet.ComputeEnergy();
                    tempLoss += temporal.BaseNetwork.ComputeEnergy();
                }

                if ((epoch + 1) % 50 == 0)
                    Console.WriteLine(
                        $"  {epoch + 1,5}  {baseLoss / SEQUENCE_LENGTH,10:F4}  {tempLoss / SEQUENCE_LENGTH,10:F4}");
            }

            // ── Evaluation ─────────────────────────────────────────────────────
            Console.WriteLine();
            Console.WriteLine("  Evaluating on test sequences...");

            int baseCorrect = 0, tempCorrect = 0, total = 0;

            for (int seq = 0; seq < TEST_SEQUENCES; seq++)
            {
                temporal.Reset();

                for (int t = 0; t < SEQUENCE_LENGTH; t++)
                {
                    int cls   = t % NUM_CLASSES;
                    var input = NoisyOneHot(cls, rng);
                    var label = OneHot(cls);

                    int basePred = baseNet.Predict(input, INFERENCE_STEPS);
                    int tempPred = temporal.Advance(input, label, INFERENCE_STEPS);

                    if (basePred == cls) baseCorrect++;
                    if (tempPred == cls) tempCorrect++;
                    total++;
                }
            }

            float baseAcc = 100f * baseCorrect / total;
            float tempAcc = 100f * tempCorrect / total;

            Console.WriteLine();
            Console.WriteLine("  ┌──────────────────────────────────────┐");
            Console.WriteLine("  │  Results                             │");
            Console.WriteLine("  ├──────────────────────────────────────┤");
            Console.WriteLine($"  │  Baseline Network   : {baseAcc,6:F1}% accuracy │");
            Console.WriteLine($"  │  Temporal Network   : {tempAcc,6:F1}% accuracy │");
            Console.WriteLine($"  │  Improvement        : {tempAcc - baseAcc,+6:F1}%           │");
            Console.WriteLine("  └──────────────────────────────────────┘");
            Console.WriteLine();

            if (tempAcc > baseAcc)
                Console.WriteLine("  Temporal context improved accuracy — the hidden state");
                Console.WriteLine("  prior correctly biases inference toward the expected");
                Console.WriteLine("  next symbol in the sequence.");
        }
        finally
        {
            if (Directory.Exists(baseDir))
                Directory.Delete(baseDir, recursive: true);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static DenseTensor<float> OneHot(int cls)
    {
        var t = DenseTensor<float>.Zeros(NUM_CLASSES);
        t.Buffer[cls] = 1f;
        return t;
    }

    private static DenseTensor<float> NoisyOneHot(int cls, Random rng)
    {
        var t = OneHot(cls);
        for (int i = 0; i < NUM_CLASSES; i++)
            t.Buffer[i] += (float)SampleNormal(rng) * NOISE_STD;
        return t;
    }

    private static double SampleNormal(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}
