using PredictiveCoding.Core;

namespace PredictiveCoding.Model;

// ─────────────────────────────────────────────────────────────────────────────
// Network  —  Hierarchical Predictive Coding network
//
// Structure:
//   Layers[0]     : sensory layer (input), state clamped to x
//   Layers[1..L-2]: hidden layers (free inference)
//   Layers[L-1]   : class/label layer (clamped to y during training)
//
// Dynamics:
//   1. INFERENCE  (multiple inference steps, Rao & Ballard style):
//      a. Top-down:  pred_l = f(W_l μ_l + b_l)   ∀l ≥ 1
//      b. Bottom-up: ε_{l-1} = μ_{l-1} − pred_l  ∀l ≥ 1
//      c. Update:    μ_l ← μ_l − η_s ∂F/∂μ_l     ∀l not clamped
//
//   2. LEARNING (after inference convergence):
//      W_l ← W_l + η_W δ_l ⊗ μ_l^T              ∀l ≥ 1
//
//   Global energy:  F = ½ Σ_{l=0}^{L-2} ||ε_l||²
//
// References:
//   Rao & Ballard (1999); Friston (2005); Whittington & Bogacz (2017)
// ─────────────────────────────────────────────────────────────────────────────

public sealed class Network
{
    // ── Architecture ─────────────────────────────────────────────────────────
    public Layer[] Layers    { get; }
    public int LayerCount => Layers.Length;

    public NetworkConfig Configuration { get; }

    public Network(NetworkConfig config, Random rng)
    {
        config.Validate();
        Configuration = config;
        Layers = BuildLayers(config.LayerSizes, rng);
    }

    // ── Layer initialisation ──────────────────────────────────────────────────

    private Layer[] BuildLayers(int[] layerSizes, Random rng)
    {
        var layers = new Layer[layerSizes.Length];

        for (int l = 0; l < layerSizes.Length; l++)
        {
            int lowerSize = l > 0 ? layerSizes[l - 1] : 0;
            // Layer l generates predictions for layer l-1
            // → W_l has shape [layerSizes[l-1], layerSizes[l]]
            var layer = new Layer
            {
                Index              = l,
                Size               = layerSizes[l],
                LowerSize          = lowerSize,
                StateLearningRate  = Configuration.StateLearningRate,
                WeightLearningRate = Configuration.WeightLearningRate,
                WeightDecay        = Configuration.WeightDecay,
                ActivationType     = Configuration.Activation,
            };
            layer.Initialize(rng);

            if (Configuration.UseDiagonalPrecision) layer.InitDiagonalPrecision();
            layers[l] = layer;
        }

        return layers;
    }

    // ── Inference ────────────────────────────────────────────────────────────

    /// <summary>
    /// Runs a full inference pass: clamps the input, optionally the top layer,
    /// then iterates multiple hidden-state update steps.
    ///
    /// Each step:
    ///   1. Top-down  : compute predictions W_l μ_l for each l ≥ 1
    ///   2. Bottom-up : compute errors ε_{l-1} = μ_{l-1} − pred_l
    ///   3. Update    : μ_l ← μ_l − η_s ∂F/∂μ_l for all non-clamped l
    /// </summary>
    public void Infer(
        DenseTensor<float>  input,
        int                 inferenceSteps,
        bool                clampTop = false,
        DenseTensor<float>? label    = null)
    {
        ClampLayer(0, Flatten(input));

        if (clampTop && label != null)
            ClampLayer(LayerCount - 1, FlattenLabel(label));
        else
            Layers[LayerCount - 1].IsClamped = false;

        for (int step = 0; step < inferenceSteps; step++)
        {
            // Top-down pass: each layer generates predictions for the layer below
            for (int l = LayerCount - 1; l >= 1; l--)
                Layers[l].ComputePrediction();

            // Bottom-up pass: ε_{l-1} = μ_{l-1} − pred_l   for l = 1..L-1
            for (int l = 1; l < LayerCount; l++)
                Layers[l - 1].SetPredictionError(Layers[l].Prediction);

            // Top layer receives no prediction from above → ε_{L-1} = 0
            Layers[LayerCount - 1].SetPredictionError(null);

            // Hidden state update: μ_l ← μ_l − η_s · (ε_l − W_l^T δ_l)
            for (int l = 1; l < LayerCount; l++)
            {
                if (Layers[l].IsClamped) continue;
                Layers[l].UpdateState(Layers[l - 1].PredictionError, Configuration.StateLearningRate);
            }
        }
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Learning step: inference with both ends clamped,
    /// then Hebbian update of the generative weights.
    ///
    ///   W_l ← W_l + η_W · δ_l ⊗ μ_l^T
    ///   b_l ← b_l + η_W · δ_l
    ///
    /// Corresponds to the learning rule of Rao &amp; Ballard (eq. 4) and
    /// the Hebbian equivalent of Whittington &amp; Bogacz (2017, eq. 20).
    /// </summary>
    public void Learn(DenseTensor<float> input, DenseTensor<float> label, int inferenceSteps)
    {
        Infer(input, inferenceSteps, clampTop: true, label: label);

        for (int l = 1; l < LayerCount; l++)
            Layers[l].UpdateWeights(Layers[l - 1].PredictionError, Configuration.WeightLearningRate);
    }

    // ── Prediction ───────────────────────────────────────────────────────────

    /// <summary>
    /// Predicts the class for an input via minimum-energy classification:
    ///
    ///   c* = argmin_c  F(x, one-hot(c))
    ///
    /// For each candidate c, fixes both the input and label = one-hot(c),
    /// converges the hidden states, and measures the residual energy.
    /// The class with the lowest energy is the most compatible with the input
    /// according to the learned generative model.
    /// (Friston 2005, "A theory of cortical responses")
    /// </summary>
    public int Predict(DenseTensor<float> input, int inferenceSteps)
    {
        int   numClasses = Layers[LayerCount - 1].Size;
        float bestEnergy = float.MaxValue;
        int   bestClass  = 0;

        var candidateLabel = DenseTensor<float>.Zeros(numClasses);

        for (int c = 0; c < numClasses; c++)
        {
            // Reset hidden states to neutral prior (zero)
            for (int l = 1; l < LayerCount - 1; l++)
                Layers[l].Mu.Zero();

            candidateLabel.Zero();
            candidateLabel.Buffer[c] = 1f;

            Infer(input, inferenceSteps, clampTop: true, label: candidateLabel);

            float energy = ComputeEnergy();
            if (energy < bestEnergy) { bestEnergy = energy; bestClass = c; }
        }

        return bestClass;
    }

    // ── Energy ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Global network energy F = ½ Σ_{l=0}^{L-2} ||ε_l||²
    ///
    /// Friston (2005) free energy for a Gaussian PC network.
    /// Decreases during inference and learning.
    /// </summary>
    public float ComputeEnergy()
    {
        float energy = 0f;
        for (int l = 0; l < LayerCount - 1; l++)
            energy += Layers[l].LocalEnergy();
        return energy;
    }

    /// <summary>Average L2 norm of the generative weights (training monitoring).</summary>
    public float AverageWeightNorm()
    {
        float sum = 0f;
        int   cnt = 0;
        for (int l = 1; l < LayerCount; l++) { sum += Layers[l].WeightNorm(); cnt++; }
        return cnt > 0 ? sum / cnt : 0f;
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    /// <summary>
    /// Returns a 1-D view of <paramref name="input"/>.
    /// If the input is already 1-D it is returned as-is (zero-copy).
    /// If it is multidimensional it is reshaped to a flat vector; the total
    /// number of elements must equal LayerSizes[0].
    /// </summary>
    private DenseTensor<float> Flatten(DenseTensor<float> input)
    {
        if (input.Rank == 1) return input;

        int expected = Configuration.LayerSizes[0];
        if (input.Length != expected)
            throw new ArgumentException(
                $"Input has {input.Length} elements but the network expects {expected} " +
                $"(LayerSizes[0]). Declared InputShape: [{string.Join(", ", Configuration.InputShape ?? [])}].");

        return TensorUtils.Reshape(input, expected);
    }

    private DenseTensor<float> FlattenLabel(DenseTensor<float> label)
    {
        if (label.Rank == 1) return label;

        int expected = Configuration.LayerSizes[^1];
        if (label.Length != expected)
            throw new ArgumentException(
                $"Label has {label.Length} elements but the network expects {expected} " +
                $"(LayerSizes[^1]). Declared OutputShape: [{string.Join(", ", Configuration.OutputShape ?? [])}].");

        return TensorUtils.Reshape(label, expected);
    }

    private void ClampLayer(int index, DenseTensor<float> value)
    {
        value.CopyTo(Layers[index].Mu);
        Layers[index].IsClamped = true;
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    public void PrintArchitecture()
    {
        Console.WriteLine("═══ Network Architecture ═══");
        for (int l = 0; l < LayerCount; l++)
        {
            var    layer = Layers[l];
            string w     = layer.GenerativeWeights != null
                ? $"W={layer.GenerativeWeights}"
                : "(no weights)";
            Console.WriteLine($"  Layer[{l}]: μ=[{layer.Size}]  {w}");
        }
        if (Configuration.InputShape is { } shape)
            Console.WriteLine($"  InputShape     : [{string.Join(", ", shape)}] → flattened to {Configuration.InputSize}");
        if (Configuration.OutputShape is { } outShape)
            Console.WriteLine($"  OutputShape    : [{string.Join(", ", outShape)}] → flattened to {Configuration.OutputSize}");
        Console.WriteLine($"  Activation: {Configuration.Activation}");
        Console.WriteLine($"  η_state={Configuration.StateLearningRate:F4}  η_weight={Configuration.WeightLearningRate:F4}  λ={Configuration.WeightDecay:G3}");
        if (Configuration.UseDiagonalPrecision) Console.WriteLine("  Diagonal precision: enabled (Huang & Rao 2011)");
        Console.WriteLine("════════════════════════════");
    }
}
