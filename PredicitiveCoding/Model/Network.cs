using PredictiveCoding.Core;

namespace PredictiveCoding.Model;

// ─────────────────────────────────────────────────────────────────────────────
// Network  —  Hierarchical Predictive Coding network
//
// Structure:
//   layer[0]     : sensory layer (input), state clamped to x
//   layer[1..L-2]: hidden layers (free inference)
//   layer[L-1]   : class/label layer (clamped to y during training)
//
// Storage:
//   All layer state lives exclusively in _store (FasterKV<int, SpanByte>).
//   Layers are read individually via LayerStore.ReadLayer(index) and written
//   back individually via LayerStore.Write(index, layer) only when modified.
//
// References:
//   Rao & Ballard (1999); Friston (2005); Whittington & Bogacz (2017)
// ─────────────────────────────────────────────────────────────────────────────

public sealed class Network : IDisposable
{
    private readonly LayerStore  _store;
    public  NetworkConfig        Configuration { get; }
    public  int                  LayerCount    => Configuration.LayerSizes.Length;

    public Network(NetworkConfig config, Random rng)
    {
        config.Validate();
        Configuration = config;
        _store = new LayerStore(config, rng, config.LazyPersistence);
    }

    // ── Inference ────────────────────────────────────────────────────────────

    /// <summary>
    /// Runs a full inference pass: clamps the input, optionally the top layer,
    /// then iterates multiple hidden-state update steps.
    /// The inferred state is transient — it is not persisted to the store.
    /// </summary>
    public void Infer(
        DenseTensor<float>  input,
        int                 inferenceSteps,
        bool                clampTop = false,
        DenseTensor<float>? label    = null)
    {
        var layers = ReadAllLayers();
        InferCore(layers, input, inferenceSteps, clampTop, label);
    }

    private void InferCore(
        Layer[]             layers,
        DenseTensor<float>  input,
        int                 inferenceSteps,
        bool                clampTop,
        DenseTensor<float>? label)
    {
        ClampLayer(layers, 0, Flatten(input));

        if (clampTop && label != null)
            ClampLayer(layers, layers.Length - 1, FlattenLabel(label));
        else
            layers[^1].IsClamped = false;

        // FFG warm-start: single top-down pass seeds hidden μ before iteration
        // (Feedforward Gist, Haider et al. 2024 — reduces inference steps needed)
        if (Configuration.UseFeedforwardGistInit)
            FeedforwardGistInit(layers);

        float actReg = Configuration.ActivityRegularization;

        for (int step = 0; step < inferenceSteps; step++)
        {
            // Top-down pass: each layer generates predictions for the layer below
            for (int l = layers.Length - 1; l >= 1; l--)
                layers[l].ComputePrediction();

            // Bottom-up pass: ε_{l-1} = μ_{l-1} − pred_l   for l = 1..L-1
            for (int l = 1; l < layers.Length; l++)
                layers[l - 1].SetPredictionError(layers[l].Prediction);

            // Top layer receives no prediction from above → ε_{L-1} = 0
            layers[^1].SetPredictionError(null);

            // Hidden state update: μ_l ← μ_l − η_s · (ε_l − W_l^T δ_l + λ_act · sign(μ_l))
            for (int l = 1; l < layers.Length; l++)
            {
                if (layers[l].IsClamped) continue;
                layers[l].UpdateState(layers[l - 1].PredictionError, Configuration.StateLearningRate, actReg);
            }
        }
    }

    /// <summary>
    /// Feedforward Gist initialisation (Haider et al. 2024, Frontiers Comput. Neurosci.).
    /// Propagates the top-layer state downward through a single forward pass,
    /// seeding each hidden layer μ_l with the top-down prediction:
    ///   μ_l ← f(W_{l+1} μ_{l+1} + b_{l+1})
    /// This warm-start is free (one forward sweep) and reduces the number of
    /// iterative inference steps needed to reach the same free-energy minimum.
    /// </summary>
    private static void FeedforwardGistInit(Layer[] layers)
    {
        for (int l = layers.Length - 1; l >= 1; l--)
        {
            layers[l].ComputePrediction();
            // Write the prediction into the layer below, but never overwrite
            // a clamped layer (layer 0 is always clamped to the input).
            if (l > 1 && !layers[l - 1].IsClamped)
                layers[l].Prediction!.CopyTo(layers[l - 1].Mu);
        }
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Learning step: inference with both ends clamped, Hebbian weight update,
    /// then the updated layers are persisted to the store.
    /// </summary>
    public void Learn(DenseTensor<float> input, DenseTensor<float> label, int inferenceSteps)
    {
        var layers = ReadAllLayers();
        InferCore(layers, input, inferenceSteps, clampTop: true, label: label);

        // Only layers with generative weights (1..L-1) are mutated.
        for (int l = 1; l < layers.Length; l++)
        {
            layers[l].UpdateWeights(layers[l - 1].PredictionError, Configuration.WeightLearningRate);
            _store.Write(l, layers[l]);
        }
    }

    // ── Prediction ───────────────────────────────────────────────────────────

    /// <summary>
    /// Predicts the class for an input via minimum-energy classification:
    ///   c* = argmin_c  F(x, one-hot(c))
    ///
    /// After returning, the store holds the network state corresponding to the
    /// winning inference so that <see cref="ComputeEnergy"/> and
    /// <see cref="ComputeLayerEnergies"/> reflect the post-inference residuals.
    /// </summary>
    public int Predict(DenseTensor<float> input, int inferenceSteps)
    {
        var   layers         = ReadAllLayers();
        int   numClasses     = layers[^1].Size;
        float bestEnergy     = float.MaxValue;
        int   bestClass      = 0;
        var   candidateLabel = DenseTensor<float>.Zeros(numClasses);
        var   flatInput      = Flatten(input);

        for (int c = 0; c < numClasses; c++)
        {
            for (int l = 1; l < layers.Length - 1; l++)
                layers[l].Mu.Zero();

            candidateLabel.Zero();
            candidateLabel.Buffer[c] = 1f;

            InferCore(layers, flatInput, inferenceSteps, clampTop: true, label: candidateLabel);

            float energy = ComputeEnergyCore(layers);
            if (energy < bestEnergy) { bestEnergy = energy; bestClass = c; }
        }

        // Re-run inference for the winning class so the persisted state reflects
        // the best explanation, not whichever candidate happened to run last.
        for (int l = 1; l < layers.Length - 1; l++)
            layers[l].Mu.Zero();
        candidateLabel.Zero();
        candidateLabel.Buffer[bestClass] = 1f;
        InferCore(layers, flatInput, inferenceSteps, clampTop: true, label: candidateLabel);

        for (int l = 0; l < layers.Length; l++)
            _store.Write(l, layers[l]);

        return bestClass;
    }

    // ── Inference trace ──────────────────────────────────────────────────────

    /// <summary>
    /// Runs inference and returns the free energy F after every step.
    /// The returned array has length <paramref name="inferenceSteps"/>.
    /// Transient: weights are not modified.
    /// </summary>
    public float[] InferTrace(DenseTensor<float> input, int inferenceSteps)
    {
        var layers   = ReadAllLayers();
        var flatInput = Flatten(input);
        ClampLayer(layers, 0, flatInput);
        layers[^1].IsClamped = false;

        if (Configuration.UseFeedforwardGistInit)
            FeedforwardGistInit(layers);

        float actReg   = Configuration.ActivityRegularization;
        var   energies = new float[inferenceSteps];

        for (int step = 0; step < inferenceSteps; step++)
        {
            for (int l = layers.Length - 1; l >= 1; l--)
                layers[l].ComputePrediction();

            for (int l = 1; l < layers.Length; l++)
                layers[l - 1].SetPredictionError(layers[l].Prediction);

            layers[^1].SetPredictionError(null);

            for (int l = 1; l < layers.Length; l++)
            {
                if (layers[l].IsClamped) continue;
                layers[l].UpdateState(layers[l - 1].PredictionError, Configuration.StateLearningRate, actReg);
            }

            energies[step] = ComputeEnergyCore(layers);
        }
        return energies;
    }

    // ── Energy ───────────────────────────────────────────────────────────────

    /// <summary>Global network energy F = ½ Σ_{l=0}^{L-2} ||ε_l||² loaded from store.</summary>
    public float ComputeEnergy()
    {
        float energy = 0f;
        for (int l = 0; l < LayerCount - 1; l++)
            energy += _store.ReadLayer(l).LocalEnergy();
        return energy;
    }

    /// <summary>Per-layer energy array [L-1], loaded from store.</summary>
    public float[] ComputeLayerEnergies()
    {
        var energies = new float[LayerCount - 1];
        for (int l = 0; l < energies.Length; l++)
            energies[l] = _store.ReadLayer(l).LocalEnergy();
        return energies;
    }

    private static float ComputeEnergyCore(Layer[] layers)
    {
        float energy = 0f;
        for (int l = 0; l < layers.Length - 1; l++)
            energy += layers[l].LocalEnergy();
        return energy;
    }

    /// <summary>Average L2 norm of the generative weights.</summary>
    public float AverageWeightNorm()
    {
        float sum = 0f;
        for (int l = 1; l < LayerCount; l++)
            sum += _store.ReadLayer(l).WeightNorm();
        return LayerCount > 1 ? sum / (LayerCount - 1) : 0f;
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    // InferCore operates on all layers simultaneously (top-down + bottom-up passes
    // in the same step), so they must all be in memory together.
    private Layer[] ReadAllLayers()
    {
        var layers = new Layer[LayerCount];
        for (int l = 0; l < LayerCount; l++)
            layers[l] = _store.ReadLayer(l);
        return layers;
    }

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

    private static void ClampLayer(Layer[] layers, int index, DenseTensor<float> value)
    {
        value.CopyTo(layers[index].Mu);
        layers[index].IsClamped = true;
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    public void PrintArchitecture()
    {
        Console.WriteLine("═══ Network Architecture ═══");
        for (int l = 0; l < LayerCount; l++)
        {
            var    layer = _store.ReadLayer(l);
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
        if (Configuration.UseDiagonalPrecision)   Console.WriteLine("  Diagonal precision:     enabled (Huang & Rao 2011)");
        if (Configuration.UseFeedforwardGistInit) Console.WriteLine("  FFG warm-start:         enabled (Haider et al. 2024)");
        if (Configuration.ActivityRegularization > 0f)
            Console.WriteLine($"  Activity regularisation: λ_act={Configuration.ActivityRegularization:G3} (L1 on μ)");
        Console.WriteLine("════════════════════════════");
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    /// <summary>
    /// Flushes any deferred writes to the persistent store.
    /// Call once per epoch when <see cref="NetworkConfig.LazyPersistence"/> is true.
    /// </summary>
    public void FlushStore() => _store.Flush();

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose() => _store.Dispose();
}
