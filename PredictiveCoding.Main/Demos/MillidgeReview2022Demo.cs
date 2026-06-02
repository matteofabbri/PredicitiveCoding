using PredictiveCoding.Core;
using PredictiveCoding.Data;
using PredictiveCoding.Logging;
using PredictiveCoding.Model;
using PredictiveCoding.Networks;
using PredictiveCoding.Training;

namespace PredictiveCoding.Main.Demos;

// ─────────────────────────────────────────────────────────────────────────────
// MillidgeReview2022Demo
//
// Five experiments inspired by:
//   Millidge, B., Seth, A., Buckley, C.L. (2022).
//   "Predictive Coding: a Theoretical and Experimental Review."
//   https://doi.org/10.48550/arXiv.2107.12979
//
//   §3  Inference   → Exp 1 (energy convergence) + Exp 2 (T ablation)
//   §4  Hierarchical PC → Exp 3 (depth) + Exp 5 (layer energies)
//   §5  Generative model  → Exp 4 (noise robustness)
//
// All experiments use 4-class Gaussian blobs in R^20 so that the demo
// runs in a few seconds without an external dataset.
// ─────────────────────────────────────────────────────────────────────────────

public static class MillidgeReview2022Demo
{
    // ── Shared dataset parameters ─────────────────────────────────────────────

    private const int InputDim      = 20;
    private const int NumClasses    = 4;
    private const int SamplesPerClass = 200;
    private const float BaseNoise   = 0.35f;
    private const int Seed          = 42;

    // ── Entry point ───────────────────────────────────────────────────────────

    public static void Run()
    {
        Banner();

        // ── Pre-train a shared reference network ──────────────────────────────
        //    Experiments 1, 2, and 5 all reuse this network so that we train
        //    only once for the common case.

        Console.WriteLine("  [Pre-training reference network for Experiments 1, 2 & 5...]");
        var (trainRef, testRef) = MakeDataset(noise: BaseNoise);
        using var refNet = TrainNetwork(
            layerSizes: [InputDim, 64, 32, NumClasses],
            trainSet: trainRef,
            epochs: 40,
            inferenceSteps: 20,
            label: "reference");

        RunExperiment1_EnergyConvergence(refNet, testRef);
        RunExperiment2_InferenceStepsAblation(refNet, testRef);
        RunExperiment3_DepthComparison();
        RunExperiment4_NoiseRobustness();
        RunExperiment5_LayerEnergyHierarchy(refNet, testRef);

        Console.WriteLine();
        Console.WriteLine("  All experiments complete.");
        Console.WriteLine(new string('═', 70));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Experiment 1 — Free-Energy Convergence During Inference  (§3)
    //
    // PC minimises the free energy F(x) = ½ Σ_l ||ε_l||²  by iterating
    // hidden-state updates.  This experiment shows that F decreases
    // monotonically with the number of steps T, confirming that the
    // inference dynamics act as gradient descent on F.
    //
    // Millidge et al. (2022) §3.1 "Inference in PC networks":
    //   "The hidden states are updated until they reach a fixed point that
    //    minimises the free energy of the system."
    // ─────────────────────────────────────────────────────────────────────────

    private static void RunExperiment1_EnergyConvergence(Network net, SyntheticDataset testSet)
    {
        Section("Experiment 1", "Free-Energy Convergence During Inference  (§3)");
        Console.WriteLine("  Each bar = average F over 40 test samples at that inference step.");
        Console.WriteLine("  Expected: monotonically decreasing — PC = gradient descent on F.");
        Console.WriteLine();

        int maxSteps  = 50;
        int nSamples  = 40;
        var sumTrace  = new float[maxSteps];
        int count     = 0;

        foreach (var batch in testSet.GetBatches(1))
        {
            if (count >= nSamples) break;
            float[] trace = net.InferTrace(batch.Inputs[0], maxSteps);
            for (int t = 0; t < maxSteps; t++) sumTrace[t] += trace[t];
            count++;
        }

        // Display at steps: 1,2,3,5,8,13,21,34,50 (Fibonacci-ish spacing)
        int[] displaySteps = [1, 2, 3, 5, 8, 13, 21, 34, 50];
        float maxF         = sumTrace.Max() / count;

        Console.WriteLine($"  {"Step",5}  Energy  Chart");
        Console.WriteLine("  " + new string('─', 55));

        foreach (int t in displaySteps)
        {
            float avg = sumTrace[t - 1] / count;
            string bar = AsciiBar(avg, maxF, width: 30);
            Console.WriteLine($"  T={t,3}   {avg,7:F3}  {bar}");
        }

        Console.WriteLine();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Experiment 2 — Inference Steps vs. Classification Accuracy  (§3)
    //
    // Millidge et al. stress that classification in PC is done by finding
    //   c* = argmin_c  F(x, one-hot(c))
    // More inference steps → closer to the fixed point → better c*.
    // ─────────────────────────────────────────────────────────────────────────

    private static void RunExperiment2_InferenceStepsAblation(Network net, SyntheticDataset testSet)
    {
        Section("Experiment 2", "Inference Steps vs. Classification Accuracy  (§3)");
        Console.WriteLine("  Classification: c* = argmin_c F(x, one-hot(c)).");
        Console.WriteLine("  More inference steps → closer to fixed point → higher accuracy.");
        Console.WriteLine();

        int[] steps = [1, 3, 5, 10, 20, 50];

        Console.WriteLine($"  {"T",5}  {"Accuracy",10}  {"Avg Energy",12}  Chart");
        Console.WriteLine("  " + new string('─', 60));

        var baselineCfg = new TrainingConfig { InferenceSteps = steps[^1] };

        foreach (int T in steps)
        {
            var cfg = baselineCfg with { InferenceSteps = T };
            var (energy, acc) = Trainer.Evaluate(net, testSet, cfg);
            string bar = AsciiBar(acc, 1f, width: 25);
            Console.WriteLine($"  T={T,3}   {acc * 100,8:F1}%   {energy,10:F3}   {bar}");
        }

        Console.WriteLine();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Experiment 3 — Network Depth Comparison  (§4)
    //
    // Deeper PC networks can represent increasingly abstract features in
    // higher layers (Millidge et al. §4: "Hierarchical Predictive Coding").
    // We compare 4 architectures with 0, 1, 2, and 3 hidden layers.
    // ─────────────────────────────────────────────────────────────────────────

    private static void RunExperiment3_DepthComparison()
    {
        Section("Experiment 3", "Network Depth Comparison  (§4 Hierarchical PC)");
        Console.WriteLine("  PC hierarchies build progressively abstract representations.");
        Console.WriteLine("  Architectures: input=20, output=4 classes, varied hidden layers.");
        Console.WriteLine();

        var architectures = new (string label, int[] layers)[]
        {
            ("No hidden",   [InputDim, NumClasses]),
            ("1 hidden",    [InputDim, 64, NumClasses]),
            ("2 hidden",    [InputDim, 64, 32, NumClasses]),
            ("3 hidden",    [InputDim, 64, 32, 16, NumClasses]),
        };

        Console.WriteLine($"  {"Depth",-12}  {"Architecture",-28}  {"Accuracy",10}  Chart");
        Console.WriteLine("  " + new string('─', 72));

        var (train, test) = MakeDataset(noise: BaseNoise);

        foreach (var (label, layers) in architectures)
        {
            using var net = TrainNetwork(layers, train, epochs: 30, inferenceSteps: 20, label);

            var cfg        = new TrainingConfig { InferenceSteps = 20 };
            var (_, acc)   = Trainer.Evaluate(net, test, cfg);
            string arch    = "[" + string.Join(", ", layers) + "]";
            string bar     = AsciiBar(acc, 1f, width: 22);
            Console.WriteLine($"  {label,-12}  {arch,-28}  {acc * 100,8:F1}%   {bar}");
        }

        Console.WriteLine();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Experiment 4 — Noise Robustness  (§5)
    //
    // PC is grounded in a generative model with an explicit noise model.
    // A network trained at a given noise level encodes prior beliefs about
    // the sensory precision.  We train once and evaluate at increasing noise
    // to reveal the degradation curve of the learned generative model.
    //
    // Millidge et al. (2022) §5: "the precision weighting of sensory evidence
    // is a core property of the predictive coding framework."
    // ─────────────────────────────────────────────────────────────────────────

    private static void RunExperiment4_NoiseRobustness()
    {
        Section("Experiment 4", "Noise Robustness — Generative Model Property  (§5)");
        Console.WriteLine($"  Network trained at noise={BaseNoise:F2}; evaluated at varying noise.");
        Console.WriteLine("  PC degrades gracefully: the generative model was fit to train noise.");
        Console.WriteLine();

        // Train once at baseline noise
        var (trainSet, _) = MakeDataset(noise: BaseNoise);
        using var net     = TrainNetwork(
            [InputDim, 64, 32, NumClasses], trainSet,
            epochs: 30, inferenceSteps: 20, "noise-robust");

        float[] noiseValues = [0.05f, 0.15f, 0.35f, 0.55f, 0.75f, 1.00f, 1.50f];

        var cfg = new TrainingConfig { InferenceSteps = 20 };

        Console.WriteLine($"  {"Noise",7}  {"Accuracy",10}  Chart");
        Console.WriteLine("  " + new string('─', 55));

        foreach (float noise in noiseValues)
        {
            var (_, testNoise) = MakeDataset(noise: noise, seed: Seed);
            var (_, acc) = Trainer.Evaluate(net, testNoise, cfg);
            string marker = Math.Abs(noise - BaseNoise) < 0.01f ? "  ← train noise" : "";
            string bar    = AsciiBar(acc, 1f, width: 25);
            Console.WriteLine($"  σ={noise,5:F2}   {acc * 100,8:F1}%   {bar}{marker}");
        }

        Console.WriteLine();
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Experiment 5 — Layer-wise Prediction Error Hierarchy  (§4)
    //
    // A key claim of hierarchical PC is that each layer l encodes a
    // prediction error ε_l = μ_l − f(W_{l+1} μ_{l+1}) that reflects
    // how well the layer above predicts the layer below.
    // After learning, lower layers carry most of the sensory residual;
    // higher layers should converge toward zero error.
    //
    // Millidge et al. (2022) §4: "higher-level representations predict
    // lower-level ones, and prediction errors propagate upward."
    // ─────────────────────────────────────────────────────────────────────────

    private static void RunExperiment5_LayerEnergyHierarchy(Network net, SyntheticDataset testSet)
    {
        Section("Experiment 5", "Layer-wise Prediction Error Hierarchy  (§4)");
        Console.WriteLine("  Average ½||ε_l||² per layer over the test set after training.");
        Console.WriteLine("  Expected: lower layers carry more residual than higher layers.");
        Console.WriteLine();

        int nSamples         = Math.Min(100, testSet.TotalSamples);
        float[]? accumulated = null;
        int count            = 0;

        foreach (var batch in testSet.GetBatches(1))
        {
            if (count >= nSamples) break;
            net.Predict(batch.Inputs[0], inferenceSteps: 20);
            float[] layerE = net.ComputeLayerEnergies();
            accumulated ??= new float[layerE.Length];
            for (int l = 0; l < layerE.Length; l++) accumulated[l] += layerE[l];
            count++;
        }

        if (accumulated is null) return;

        float maxVal = accumulated.Max() / count;

        Console.WriteLine($"  {"Layer",8}  {"Avg F_l",10}  Chart");
        Console.WriteLine("  " + new string('─', 55));

        string[] labels = ["sensory", "hidden-1", "hidden-2", "output"];

        for (int l = 0; l < accumulated.Length; l++)
        {
            float avg   = accumulated[l] / count;
            string lbl  = l < labels.Length ? labels[l] : $"layer-{l}";
            string bar  = AsciiBar(avg, maxVal, width: 28);
            Console.WriteLine($"  ε[{l}] {lbl,-9}  {avg,10:F4}  {bar}");
        }

        Console.WriteLine();
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private static (SyntheticDataset train, SyntheticDataset test) MakeDataset(
        float noise, int seed = Seed) =>
        SyntheticDataset.Split(
            numClasses    : NumClasses,
            inputDim      : InputDim,
            totalPerClass : SamplesPerClass,
            testFraction  : 0.2f,
            noise         : noise,
            seed          : seed,
            verbose       : false);

    private static Network TrainNetwork(
        int[]  layerSizes,
        IDataset trainSet,
        int    epochs,
        int    inferenceSteps,
        string label)
    {
        string dir = Path.Combine(Path.GetTempPath(), $"pcmillidge_{Guid.NewGuid():N}");

        var cfg = new TrainingConfig
        {
            Network = new NetworkConfig
            {
                Path               = dir,
                LayerSizes         = layerSizes,
                Activation         = ActivationType.Tanh,
                StateLearningRate  = 0.05f,
                WeightLearningRate = 0.002f,
            },
            InferenceSteps  = inferenceSteps,
            Epochs          = epochs,
            BatchSize       = 16,
            Seed            = Seed,
            LogEveryNEpochs = int.MaxValue,  // suppress per-epoch output
        };

        var rng     = new Random(Seed);
        var net     = new Network(cfg.Network, rng);
        var logger  = new MetricLogger(verbose: false);
        var dummyTest = trainSet;  // training only; no test evaluation overhead

        Console.Write($"    Training [{string.Join(", ", layerSizes)}] × {epochs} epochs ...");

        // Minimal training loop (no evaluation in the loop for speed).
        var trainRng = new Random(Seed);
        for (int epoch = 1; epoch <= epochs; epoch++)
            Trainer.TrainEpoch(net, trainSet, cfg, trainRng);

        Console.WriteLine(" done.");

        return net;
    }

    private static string AsciiBar(float value, float maxValue, int width)
    {
        if (maxValue <= 0f) return new string('░', width);
        int filled = (int)Math.Round(Math.Clamp(value / maxValue, 0f, 1f) * width);
        return new string('█', filled) + new string('░', width - filled);
    }

    private static void Section(string id, string title)
    {
        Console.WriteLine();
        Console.WriteLine(new string('─', 70));
        Console.WriteLine($"  {id}  —  {title}");
        Console.WriteLine(new string('─', 70));
    }

    private static void Banner()
    {
        Console.WriteLine();
        Console.WriteLine(new string('═', 70));
        Console.WriteLine("  Predictive Coding — Experiments from Millidge, Seth & Buckley (2022)");
        Console.WriteLine("  \"Predictive Coding: a Theoretical and Experimental Review\"");
        Console.WriteLine("  https://doi.org/10.48550/arXiv.2107.12979");
        Console.WriteLine(new string('═', 70));
        Console.WriteLine();
        Console.WriteLine("  Dataset: 4-class Gaussian blobs in R^20");
        Console.WriteLine($"  Base noise: σ={BaseNoise:F2},  {SamplesPerClass} samples/class");
        Console.WriteLine();
    }
}
