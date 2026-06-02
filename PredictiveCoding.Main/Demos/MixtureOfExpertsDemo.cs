using PredictiveCoding.Core;
using PredictiveCoding.Model;

namespace PredictiveCoding.Main.Demos;

// ─────────────────────────────────────────────────────────────────────────────
// MixtureOfExpertsDemo
//
// Demonstrates expert specialisation in a bimodal classification task.
//
// Task:
//   4 classes, two distinct input "modes" per class:
//     Mode A — active dimensions 0..3  (e.g. visual feature set A)
//     Mode B — active dimensions 4..7  (e.g. visual feature set B)
//   Both modes share the same label space (same 4 classes).
//
// Hypothesis:
//   A single Network must fit both modes with one set of weights, producing
//   a mediocre average.  A MoE with 2+ experts allows specialisation:
//   one expert captures Mode A, another Mode B.  Gate weights should reflect
//   which mode the current input belongs to, and total accuracy should exceed
//   the single-network baseline.
//
// Outputs:
//   - Training energy per epoch (baseline vs MoE)
//   - Test accuracy (baseline vs MoE)
//   - Average gate weight per expert, broken down by mode
//   - Expert weight norms (proxy for specialisation degree)
// ─────────────────────────────────────────────────────────────────────────────

public static class MixtureOfExpertsDemo
{
    private const int   NUM_CLASSES        = 4;
    private const int   INPUT_DIM          = 8;   // dims 0-3 = mode A, 4-7 = mode B
    private const int   HIDDEN_SIZE        = 32;
    private const int   NUM_EXPERTS        = 2;
    private const int   TRAIN_SAMPLES      = 2000;
    private const int   TEST_SAMPLES       = 400;
    private const int   INFERENCE_STEPS    = 20;
    private const float STATE_LR           = 0.1f;
    private const float WEIGHT_LR          = 0.005f;
    private const float NOISE_STD          = 0.25f;
    private const float GATING_TEMPERATURE = 2f;

    public static void Run()
    {
        Console.WriteLine();
        Console.WriteLine("╔══════════════════════════════════════════════════════╗");
        Console.WriteLine("║         Mixture of Experts (PC energy gating)        ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════╝");
        Console.WriteLine();
        Console.WriteLine($"  Task     : {NUM_CLASSES} classes × 2 input modes (bimodal)");
        Console.WriteLine($"  Experts  : {NUM_EXPERTS}  β={GATING_TEMPERATURE}");
        Console.WriteLine($"  Noise    : σ={NOISE_STD}");
        Console.WriteLine();

        var rng     = new Random(42);
        string root = Path.Combine(Path.GetTempPath(), $"moe_{Guid.NewGuid():N}");

        try
        {
            var netCfg = new NetworkConfig
            {
                LayerSizes         = [INPUT_DIM, HIDDEN_SIZE, NUM_CLASSES],
                Activation         = ActivationType.Tanh,
                StateLearningRate  = STATE_LR,
                WeightLearningRate = WEIGHT_LR,
            };

            using var baseline = new Network(netCfg with { Path = Path.Combine(root, "base") }, new Random(1));

            var moeCfg = new MixtureOfExpertsConfig
            {
                BasePath           = Path.Combine(root, "moe"),
                ExpertCount        = NUM_EXPERTS,
                ExpertConfig       = netCfg,
                GatingTemperature  = GATING_TEMPERATURE,
            };
            using var moe = new MixtureOfExperts(moeCfg, new Random(1));

            // ── Training ──────────────────────────────────────────────────────
            Console.WriteLine($"  Training on {TRAIN_SAMPLES} samples...");
            Console.WriteLine();
            Console.WriteLine($"  {"Step",6}  {"Base energy",12}  {"MoE energy",12}  {"Gates",20}");
            Console.WriteLine("  " + new string('─', 58));

            float sumBaseE = 0f;
            float[] sumMoeG = new float[NUM_EXPERTS];

            for (int step = 0; step < TRAIN_SAMPLES; step++)
            {
                bool modeA = (step % 2 == 0);           // alternate modes
                int  cls   = rng.Next(NUM_CLASSES);
                var  input = GenerateInput(cls, modeA, rng);
                var  label = OneHot(cls);

                baseline.Learn(input, label, INFERENCE_STEPS);
                moe.Learn(input, label, INFERENCE_STEPS);

                sumBaseE += baseline.ComputeEnergy();
                for (int i = 0; i < NUM_EXPERTS; i++)
                    sumMoeG[i] += moe.LastGateWeights[i];

                if ((step + 1) % 500 == 0)
                {
                    float avgBase = sumBaseE / 500f;
                    float avgMoeE = moe.LastExpertEnergies.Min();
                    string gates  = string.Join("  ", sumMoeG.Select((g, i) => $"g{i}={g / 500f:F2}"));
                    Console.WriteLine($"  {step + 1,6}  {avgBase,12:F4}  {avgMoeE,12:F4}  {gates,20}");
                    sumBaseE = 0f;
                    Array.Clear(sumMoeG);
                }
            }

            // ── Evaluation ────────────────────────────────────────────────────
            Console.WriteLine();
            Console.WriteLine($"  Evaluating on {TEST_SAMPLES} test samples...");

            int baseCorrect = 0, moeCorrect = 0;
            var gateByMode  = new float[2, NUM_EXPERTS];  // [mode, expert]
            var gateCountByMode = new int[2];

            for (int t = 0; t < TEST_SAMPLES; t++)
            {
                bool modeA = (t % 2 == 0);
                int  cls   = rng.Next(NUM_CLASSES);
                var  input = GenerateInput(cls, modeA, rng);
                var  label = OneHot(cls);

                int basePred = baseline.Predict(input, INFERENCE_STEPS);
                var moeResult = moe.PredictDetailed(input, INFERENCE_STEPS);

                if (basePred          == cls) baseCorrect++;
                if (moeResult.PredictedClass == cls) moeCorrect++;

                // Accumulate gate weights to measure specialisation
                moe.Learn(input, label, INFERENCE_STEPS);   // one more step for gate stats
                int modeIdx = modeA ? 0 : 1;
                for (int i = 0; i < NUM_EXPERTS; i++)
                    gateByMode[modeIdx, i] += moe.LastGateWeights[i];
                gateCountByMode[modeIdx]++;
            }

            float baseAcc = 100f * baseCorrect / TEST_SAMPLES;
            float moeAcc  = 100f * moeCorrect  / TEST_SAMPLES;

            // ── Results ────────────────────────────────────────────────────────
            Console.WriteLine();
            Console.WriteLine("  ┌────────────────────────────────────────────────────┐");
            Console.WriteLine("  │  Accuracy                                          │");
            Console.WriteLine("  ├────────────────────────────────────────────────────┤");
            Console.WriteLine($"  │  Baseline (single network) : {baseAcc,6:F1}%               │");
            Console.WriteLine($"  │  Mixture of Experts        : {moeAcc,6:F1}%               │");
            Console.WriteLine($"  │  Improvement               : {moeAcc - baseAcc,+6:F1}%               │");
            Console.WriteLine("  └────────────────────────────────────────────────────┘");
            Console.WriteLine();

            Console.WriteLine("  Expert specialisation (average gate weight per mode):");
            Console.WriteLine($"  {"",12}  {string.Join("  ", Enumerable.Range(0, NUM_EXPERTS).Select(i => $"Expert {i}"))}");
            string[] modeNames = ["Mode A (dim 0-3)", "Mode B (dim 4-7)"];
            for (int m = 0; m < 2; m++)
            {
                var gateStr = string.Join("  ", Enumerable.Range(0, NUM_EXPERTS)
                    .Select(i => $"{gateByMode[m, i] / gateCountByMode[m],8:F3}"));
                Console.WriteLine($"  {modeNames[m],16}: {gateStr}");
            }

            Console.WriteLine();
            Console.WriteLine("  Expert weight norms:");
            var norms = moe.ExpertWeightNorms();
            for (int i = 0; i < NUM_EXPERTS; i++)
                Console.WriteLine($"    Expert {i}: {norms[i]:F4}");
        }
        finally
        {
            if (Directory.Exists(root))
                Directory.Delete(root, recursive: true);
        }
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static DenseTensor<float> GenerateInput(int cls, bool modeA, Random rng)
    {
        // Mode A: signal in dims 0-3 (one-hot at position cls)
        // Mode B: signal in dims 4-7 (one-hot at position cls + 4)
        var t = DenseTensor<float>.Zeros(INPUT_DIM);
        int activeIdx = modeA ? cls : cls + NUM_CLASSES;
        t.Buffer[activeIdx] = 1f;

        // Add noise across all dimensions
        for (int i = 0; i < INPUT_DIM; i++)
            t.Buffer[i] += (float)SampleNormal(rng) * NOISE_STD;

        return t;
    }

    private static DenseTensor<float> OneHot(int cls)
    {
        var t = DenseTensor<float>.Zeros(NUM_CLASSES);
        t.Buffer[cls] = 1f;
        return t;
    }

    private static double SampleNormal(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Sin(2.0 * Math.PI * u2);
    }
}
