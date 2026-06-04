using PredictiveCoding.Core;
using PredictiveCoding.Model;

namespace PredictiveCoding.Networks;

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
    private readonly LayerStore             _store;
    private readonly ReaderWriterLockSlim   _rwLock = new(LockRecursionPolicy.SupportsRecursion);
    public  NetworkConfig                   Configuration { get; }
    public  int                             LayerCount    => Configuration.LayerSizes.Length;

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
    public void Infer(DenseTensor<float>  input, int inferenceSteps, bool clampTop = false, DenseTensor<float>? label = null)
    {
        _rwLock.EnterReadLock();
        try
        {
            var layers = ReadAllLayers();
            InferCore(layers, input, inferenceSteps, clampTop, label);
        }
        finally { _rwLock.ExitReadLock(); }
    }

    private void InferCore(Layer[] layers, DenseTensor<float>  input, int inferenceSteps, bool clampTop, DenseTensor<float>? label)
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
        _rwLock.EnterWriteLock();
        try
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
        finally { _rwLock.ExitWriteLock(); }
    }

    /// <summary>
    /// Gated learning step for Mixture-of-Experts: identical to
    /// <see cref="Learn"/> but scales the effective weight learning rate by
    /// <paramref name="gateWeight"/> (the responsibility assigned to this expert
    /// by the MoE gating network).
    ///
    /// Returns the free energy <c>F</c> after inference so that the caller can
    /// use it to recompute gate weights if needed.
    ///
    /// When <paramref name="gateWeight"/> == 0 the weight update is skipped
    /// entirely (pure energy probe).
    /// </summary>
    public float LearnGated(
        DenseTensor<float> input,
        DenseTensor<float> label,
        int                inferenceSteps,
        float              gateWeight)
    {
        _rwLock.EnterWriteLock();
        try
        {
            var   layers = ReadAllLayers();
            InferCore(layers, input, inferenceSteps, clampTop: true, label: label);
            float energy = ComputeEnergyCore(layers);

            if (gateWeight > 0f)
            {
                float effectiveLr = Configuration.WeightLearningRate * gateWeight;
                for (int l = 1; l < layers.Length; l++)
                {
                    layers[l].UpdateWeights(layers[l - 1].PredictionError, effectiveLr);
                    _store.Write(l, layers[l]);
                }
            }

            return energy;
        }
        finally { _rwLock.ExitWriteLock(); }
    }

    /// <summary>
    /// Runs inference with both ends clamped and returns the resulting free
    /// energy without modifying any weights.  Used by
    /// <c>MixtureOfExperts</c> to compute per-expert gate weights before
    /// deciding how much each expert should learn.
    /// </summary>
    public float InferEnergy(DenseTensor<float> input, DenseTensor<float> label, int inferenceSteps)
    {
        _rwLock.EnterReadLock();
        try
        {
            var layers = ReadAllLayers();
            InferCore(layers, input, inferenceSteps, clampTop: true, label: label);
            return ComputeEnergyCore(layers);
        }
        finally { _rwLock.ExitReadLock(); }
    }

    /// <summary>
    /// Returns the free energy F(x, one-hot(c)) for every class c.
    /// Length of the returned array equals the number of output units.
    /// Transient: weights are not modified.
    /// Used by <c>EnsembleNetwork</c> to aggregate probability
    /// distributions across members.
    /// </summary>
    public float[] ClassEnergies(DenseTensor<float> input, int inferenceSteps)
    {
        _rwLock.EnterReadLock();
        try
        {
            var   layers         = ReadAllLayers();
            int   numClasses     = layers[^1].Size;
            var   energies       = new float[numClasses];
            var   candidateLabel = DenseTensor<float>.Zeros(numClasses);
            var   flatInput      = Flatten(input);

            for (int c = 0; c < numClasses; c++)
            {
                for (int l = 1; l < layers.Length - 1; l++)
                    layers[l].Mu.Zero();

                candidateLabel.Zero();
                candidateLabel.Buffer[c] = 1f;
                InferCore(layers, flatInput, inferenceSteps, clampTop: true, label: candidateLabel);
                energies[c] = ComputeEnergyCore(layers);
            }

            return energies;
        }
        finally { _rwLock.ExitReadLock(); }
    }

    /// <summary>
    /// Minimum-energy classification that also returns the winning free energy.
    /// Equivalent to <see cref="Predict"/> followed by <see cref="ComputeEnergy"/>
    /// but avoids the extra store round-trip.
    /// </summary>
    public (int cls, float energy) PredictWithEnergy(DenseTensor<float> input, int inferenceSteps)
    {
        _rwLock.EnterWriteLock();
        try   { return PredictCore(input, inferenceSteps); }
        finally { _rwLock.ExitWriteLock(); }
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
        _rwLock.EnterWriteLock();
        try   { return PredictCore(input, inferenceSteps).cls; }
        finally { _rwLock.ExitWriteLock(); }
    }

    // Caller must hold _rwLock (write).
    private (int cls, float energy) PredictCore(DenseTensor<float> input, int inferenceSteps)
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

        float winnerEnergy = ComputeEnergyCore(layers);
        for (int l = 0; l < layers.Length; l++)
            _store.Write(l, layers[l]);

        return (bestClass, winnerEnergy);
    }

    // ── Temporal inference & learning ────────────────────────────────────────

    /// <summary>
    /// Temporal learning step: same as <see cref="Learn"/> but injects the
    /// hidden states from the previous timestep as temporal priors before
    /// running inference (Friston 2008 dynamic causal modelling).
    ///
    /// Returns the hidden states <c>μ[1..L-2]</c> after inference so that
    /// the caller (typically <c>TemporalNetwork</c>) can pass them as
    /// priors on the next call.
    /// </summary>
    /// <param name="temporalPriors">
    /// Hidden states from the previous timestep (length = LayerCount − 2).
    /// Null on the first timestep of a new sequence.
    /// </param>
    /// <param name="temporalPrecision">
    /// Π_t: strength of the continuity gradient toward the previous state.
    /// </param>
    public DenseTensor<float>[] LearnTemporal(
        DenseTensor<float>    input,
        DenseTensor<float>    label,
        int                   inferenceSteps,
        DenseTensor<float>[]? temporalPriors,
        float                 temporalPrecision)
    {
        _rwLock.EnterWriteLock();
        try
        {
            var layers = ReadAllLayers();
            if (temporalPriors != null)
                InjectTemporalPriors(layers, temporalPriors, temporalPrecision);

            InferCore(layers, input, inferenceSteps, clampTop: true, label: label);

            for (int l = 1; l < layers.Length; l++)
            {
                layers[l].UpdateWeights(layers[l - 1].PredictionError, Configuration.WeightLearningRate);
                _store.Write(l, layers[l]);
            }

            return ExtractHiddenStates(layers);
        }
        finally { _rwLock.ExitWriteLock(); }
    }

    /// <summary>
    /// Temporal prediction: minimum-energy classification conditioned on the
    /// hidden states from the previous timestep.
    /// </summary>
    public int PredictTemporal(
        DenseTensor<float>    input,
        int                   inferenceSteps,
        DenseTensor<float>[]? temporalPriors,
        float                 temporalPrecision)
    {
        _rwLock.EnterWriteLock();
        try
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

                if (temporalPriors != null)
                    InjectTemporalPriors(layers, temporalPriors, temporalPrecision);

                candidateLabel.Zero();
                candidateLabel.Buffer[c] = 1f;

                InferCore(layers, flatInput, inferenceSteps, clampTop: true, label: candidateLabel);

                float energy = ComputeEnergyCore(layers);
                if (energy < bestEnergy) { bestEnergy = energy; bestClass = c; }
            }

            // Re-run for winner so persisted state reflects the best explanation.
            for (int l = 1; l < layers.Length - 1; l++)
                layers[l].Mu.Zero();

            if (temporalPriors != null)
                InjectTemporalPriors(layers, temporalPriors, temporalPrecision);

            candidateLabel.Zero();
            candidateLabel.Buffer[bestClass] = 1f;
            InferCore(layers, flatInput, inferenceSteps, clampTop: true, label: candidateLabel);

            for (int l = 0; l < layers.Length; l++)
                _store.Write(l, layers[l]);

            return bestClass;
        }
        finally { _rwLock.ExitWriteLock(); }
    }

    /// <summary>
    /// Runs inference with optional temporal priors and returns the hidden
    /// states after convergence.  Transient: weights are not modified.
    /// </summary>
    public DenseTensor<float>[] InferTemporal(
        DenseTensor<float>    input,
        int                   inferenceSteps,
        DenseTensor<float>[]? temporalPriors,
        float                 temporalPrecision,
        bool                  clampTop = false,
        DenseTensor<float>?   label    = null)
    {
        _rwLock.EnterReadLock();
        try
        {
            var layers = ReadAllLayers();
            if (temporalPriors != null)
                InjectTemporalPriors(layers, temporalPriors, temporalPrecision);
            InferCore(layers, input, inferenceSteps, clampTop, label);
            return ExtractHiddenStates(layers);
        }
        finally { _rwLock.ExitReadLock(); }
    }

    /// <summary>
    /// Probes the temporal free energy without updating any weights.
    /// Equivalent to <see cref="InferEnergy"/> but injects temporal priors first.
    /// Used by <c>TemporalMixtureOfExperts</c> to compute per-expert gate
    /// weights before deciding how much each expert should learn.
    /// </summary>
    public float InferEnergyTemporal(
        DenseTensor<float>    input,
        DenseTensor<float>    label,
        int                   inferenceSteps,
        DenseTensor<float>[]? temporalPriors,
        float                 temporalPrecision)
    {
        _rwLock.EnterReadLock();
        try
        {
            var layers = ReadAllLayers();
            if (temporalPriors != null)
                InjectTemporalPriors(layers, temporalPriors, temporalPrecision);
            InferCore(layers, input, inferenceSteps, clampTop: true, label: label);
            return ComputeEnergyCore(layers);
        }
        finally { _rwLock.ExitReadLock(); }
    }

    /// <summary>
    /// Gated temporal learning step for <c>TemporalMixtureOfExperts</c>.
    /// Injects temporal priors, runs inference, then scales the weight update by
    /// <paramref name="gateWeight"/>.  When <paramref name="gateWeight"/> == 0 the
    /// weight update is skipped (pure temporal context advance).
    /// Returns the post-inference hidden states and free energy.
    /// </summary>
    public (DenseTensor<float>[] hiddenStates, float energy) LearnTemporalGated(
        DenseTensor<float>    input,
        DenseTensor<float>    label,
        int                   inferenceSteps,
        DenseTensor<float>[]? temporalPriors,
        float                 temporalPrecision,
        float                 gateWeight)
    {
        _rwLock.EnterWriteLock();
        try
        {
            var layers = ReadAllLayers();
            if (temporalPriors != null)
                InjectTemporalPriors(layers, temporalPriors, temporalPrecision);
            InferCore(layers, input, inferenceSteps, clampTop: true, label: label);
            float energy = ComputeEnergyCore(layers);

            if (gateWeight > 0f)
            {
                float effectiveLr = Configuration.WeightLearningRate * gateWeight;
                for (int l = 1; l < layers.Length; l++)
                {
                    layers[l].UpdateWeights(layers[l - 1].PredictionError, effectiveLr);
                    _store.Write(l, layers[l]);
                }
            }

            return (ExtractHiddenStates(layers), energy);
        }
        finally { _rwLock.ExitWriteLock(); }
    }

    /// <summary>
    /// Temporal minimum-energy classification that also returns the winning free
    /// energy.  Combines <see cref="PredictTemporal"/> and energy reporting in a
    /// single write-lock pass.
    /// </summary>
    public (int cls, float energy) PredictWithEnergyTemporal(
        DenseTensor<float>    input,
        int                   inferenceSteps,
        DenseTensor<float>[]? temporalPriors,
        float                 temporalPrecision)
    {
        _rwLock.EnterWriteLock();
        try
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

                if (temporalPriors != null)
                    InjectTemporalPriors(layers, temporalPriors, temporalPrecision);

                candidateLabel.Zero();
                candidateLabel.Buffer[c] = 1f;
                InferCore(layers, flatInput, inferenceSteps, clampTop: true, label: candidateLabel);

                float energy = ComputeEnergyCore(layers);
                if (energy < bestEnergy) { bestEnergy = energy; bestClass = c; }
            }

            // Re-run for winner so persisted state reflects the best explanation.
            for (int l = 1; l < layers.Length - 1; l++)
                layers[l].Mu.Zero();

            if (temporalPriors != null)
                InjectTemporalPriors(layers, temporalPriors, temporalPrecision);

            candidateLabel.Zero();
            candidateLabel.Buffer[bestClass] = 1f;
            InferCore(layers, flatInput, inferenceSteps, clampTop: true, label: candidateLabel);

            float winnerEnergy = ComputeEnergyCore(layers);
            for (int l = 0; l < layers.Length; l++)
                _store.Write(l, layers[l]);

            return (bestClass, winnerEnergy);
        }
        finally { _rwLock.ExitWriteLock(); }
    }

    // ── Temporal helpers ─────────────────────────────────────────────────────

    private static void InjectTemporalPriors(
        Layer[]               layers,
        DenseTensor<float>[]  priors,
        float                 precision)
    {
        // priors covers hidden layers 1..L-2  (length = L-2)
        int hiddenCount = layers.Length - 2;
        if (priors.Length != hiddenCount)
            throw new ArgumentException(
                $"Expected {hiddenCount} temporal prior(s) (one per hidden layer) " +
                $"but got {priors.Length}.");

        for (int i = 0; i < hiddenCount; i++)
        {
            layers[i + 1].TemporalPrior     = priors[i];
            layers[i + 1].TemporalPrecision = precision;
        }
    }

    private static DenseTensor<float>[] ExtractHiddenStates(Layer[] layers)
    {
        int hiddenCount = layers.Length - 2;
        var states = new DenseTensor<float>[hiddenCount];
        for (int i = 0; i < hiddenCount; i++)
            states[i] = layers[i + 1].Mu.Clone();
        return states;
    }

    // ── Inference trace ──────────────────────────────────────────────────────

    /// <summary>
    /// Runs inference and returns the free energy F after every step.
    /// The returned array has length <paramref name="inferenceSteps"/>.
    /// Transient: weights are not modified.
    /// </summary>
    public float[] InferTrace(DenseTensor<float> input, int inferenceSteps)
    {
        _rwLock.EnterReadLock();
        try { return InferTraceCore(input, inferenceSteps); }
        finally { _rwLock.ExitReadLock(); }
    }

    private float[] InferTraceCore(DenseTensor<float> input, int inferenceSteps)
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
        _rwLock.EnterReadLock();
        try
        {
            float energy = 0f;
            for (int l = 0; l < LayerCount - 1; l++)
                energy += _store.ReadLayer(l).LocalEnergy();
            return energy;
        }
        finally { _rwLock.ExitReadLock(); }
    }

    /// <summary>Per-layer energy array [L-1], loaded from store.</summary>
    public float[] ComputeLayerEnergies()
    {
        _rwLock.EnterReadLock();
        try
        {
            var energies = new float[LayerCount - 1];
            for (int l = 0; l < energies.Length; l++)
                energies[l] = _store.ReadLayer(l).LocalEnergy();
            return energies;
        }
        finally { _rwLock.ExitReadLock(); }
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
        _rwLock.EnterReadLock();
        try
        {
            float sum = 0f;
            for (int l = 1; l < LayerCount; l++)
                sum += _store.ReadLayer(l).WeightNorm();
            return LayerCount > 1 ? sum / (LayerCount - 1) : 0f;
        }
        finally { _rwLock.ExitReadLock(); }
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
        int expected = Configuration.LayerSizes[0];

        if (input.Rank == 1)
        {
            if (input.Length != expected)
                throw new ArgumentException(
                    $"Input has {input.Length} elements but the network expects {expected} " +
                    $"(LayerSizes[0]). Declared InputShape: [{string.Join(", ", Configuration.InputShape ?? [])}].");
            return input;
        }

        if (input.Length != expected)
            throw new ArgumentException(
                $"Input has {input.Length} elements but the network expects {expected} " +
                $"(LayerSizes[0]). Declared InputShape: [{string.Join(", ", Configuration.InputShape ?? [])}].");

        return TensorUtils.Reshape(input, expected);
    }

    private DenseTensor<float> FlattenLabel(DenseTensor<float> label)
    {
        int expected = Configuration.LayerSizes[^1];

        if (label.Rank == 1)
        {
            if (label.Length != expected)
                throw new ArgumentException(
                    $"Label has {label.Length} elements but the network expects {expected} " +
                    $"(LayerSizes[^1]). Declared OutputShape: [{string.Join(", ", Configuration.OutputShape ?? [])}].");
            return label;
        }

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
    public void FlushStore()
    {
        _rwLock.EnterWriteLock();
        try   { _store.Flush(); }
        finally { _rwLock.ExitWriteLock(); }
    }

    // ── Generative / Brain support ────────────────────────────────────────────

    /// <summary>
    /// Runs free inference (top layer unclamped) and returns a clone of
    /// μ_{L-1} after convergence.  Used as an encoder by
    /// <c>Expanded.GenerativeNetwork</c> and <c>Expanded.MultiModalNetwork</c>.
    /// Transient: weights are not modified.
    /// </summary>
    public DenseTensor<float> InferGetTop(DenseTensor<float> input, int inferenceSteps)
    {
        _rwLock.EnterReadLock();
        try
        {
            var layers = ReadAllLayers();
            InferCore(layers, input, inferenceSteps, clampTop: false, label: null);
            return layers[^1].Mu.Clone();
        }
        finally { _rwLock.ExitReadLock(); }
    }

    /// <summary>
    /// Clamps the TOP layer to <paramref name="latent"/> and runs inference with
    /// layer[0] unconstrained, returning layer[0].Mu as the generated sample.
    /// This is the generative (decoding) direction of the PC network:
    ///   μ_0 ← argmin_{μ_0} F( latent fixed, μ_0 free )
    /// Transient: weights are not modified.
    /// </summary>
    public DenseTensor<float> Generate(DenseTensor<float> latent, int inferenceSteps)
    {
        int expectedLatentSize = Configuration.LayerSizes[^1];
        if (latent.Length != expectedLatentSize)
            throw new ArgumentException(
                $"Latent vector has {latent.Length} elements; top layer expects {expectedLatentSize}.");

        _rwLock.EnterReadLock();
        try
        {
            var layers = ReadAllLayers();
            return GenerateCore(layers, latent, inferenceSteps);
        }
        finally { _rwLock.ExitReadLock(); }
    }

    private DenseTensor<float> GenerateCore(Layer[] layers, DenseTensor<float> latent, int inferenceSteps)
    {
        ClampLayer(layers, layers.Length - 1, latent);

        // Layer[0] is free (to be generated) — start from zero.
        layers[0].Mu.Zero();
        layers[0].IsClamped = false;

        float etaState = Configuration.StateLearningRate;
        float actReg   = Configuration.ActivityRegularization;

        for (int step = 0; step < inferenceSteps; step++)
        {
            for (int l = layers.Length - 1; l >= 1; l--)
                layers[l].ComputePrediction();

            for (int l = 1; l < layers.Length; l++)
                layers[l - 1].SetPredictionError(layers[l].Prediction);

            // Top layer is clamped; its error is unused but set for consistency.
            layers[^1].SetPredictionError(null);

            // Update hidden layers [1..L-2] via the standard PC rule.
            for (int l = 1; l < layers.Length - 1; l++)
            {
                if (layers[l].IsClamped) continue;
                layers[l].UpdateState(layers[l - 1].PredictionError, etaState, actReg);
            }

            // Layer[0] has no generative weights so UpdateState skips it.
            // Apply the raw gradient: μ_0 -= η * ε_0
            if (!layers[0].IsClamped)
                TensorUtils.InPlaceAxpy(-etaState, layers[0].PredictionError, layers[0].Mu);
        }

        return layers[0].Mu.Clone();
    }

    /// <summary>
    /// Unsupervised learning step: inference without clamping the top layer,
    /// followed by a Hebbian weight update.  If the top layer has a
    /// <see cref="Model.Layer.PriorMean"/> set it regularises the latent toward
    /// the prior (e.g. N(0, I) for a generative model).
    /// Persists updated weights to the store.
    /// </summary>
    public void LearnUnsupervised(DenseTensor<float> input, int inferenceSteps)
    {
        _rwLock.EnterWriteLock();
        try
        {
            var layers = ReadAllLayers();
            InferCore(layers, input, inferenceSteps, clampTop: false, label: null);
            for (int l = 1; l < layers.Length; l++)
            {
                layers[l].UpdateWeights(layers[l - 1].PredictionError, Configuration.WeightLearningRate);
                _store.Write(l, layers[l]);
            }
        }
        finally { _rwLock.ExitWriteLock(); }
    }

    /// <summary>
    /// Temporal unsupervised learning step (encoder + dynamics): injects the
    /// previous timestep's hidden states as temporal priors, runs inference with
    /// the top layer FREE (no label clamp), updates all generative weights via the
    /// Hebbian rule, and returns both the converged top latent μ_{L-1} and the new
    /// hidden states for the next timestep.
    ///
    /// This is the temporal counterpart of <see cref="LearnUnsupervised"/> and the
    /// unsupervised counterpart of <see cref="LearnTemporal"/>.  It is the building
    /// block for <c>Expanded.DeepRecurrent.DeepRecurrentNetwork</c>, where each stage
    /// must (a) learn to encode its input, (b) carry temporal context across steps,
    /// and (c) emit a top latent that feeds the stage above.
    /// </summary>
    public (DenseTensor<float> topLatent, DenseTensor<float>[] hiddenStates) LearnUnsupervisedTemporal(
        DenseTensor<float>    input,
        int                   inferenceSteps,
        DenseTensor<float>[]? temporalPriors,
        float                 temporalPrecision)
    {
        _rwLock.EnterWriteLock();
        try
        {
            var layers = ReadAllLayers();
            if (temporalPriors != null)
                InjectTemporalPriors(layers, temporalPriors, temporalPrecision);

            InferCore(layers, input, inferenceSteps, clampTop: false, label: null);

            for (int l = 1; l < layers.Length; l++)
            {
                layers[l].UpdateWeights(layers[l - 1].PredictionError, Configuration.WeightLearningRate);
                _store.Write(l, layers[l]);
            }

            return (layers[^1].Mu.Clone(), ExtractHiddenStates(layers));
        }
        finally { _rwLock.ExitWriteLock(); }
    }

    /// <summary>
    /// Temporal encoder probe: injects the previous timestep's hidden states as
    /// temporal priors, runs free inference (top layer unclamped) and returns the
    /// converged top latent μ_{L-1} together with the new hidden states.
    /// Transient — no weights are modified.  Used by
    /// <c>Expanded.DeepRecurrent.DeepRecurrentNetwork</c> during evaluation to roll
    /// the temporal context forward while feeding latents up the cascade.
    /// </summary>
    public (DenseTensor<float> topLatent, DenseTensor<float>[] hiddenStates) InferGetTopTemporal(
        DenseTensor<float>    input,
        int                   inferenceSteps,
        DenseTensor<float>[]? temporalPriors,
        float                 temporalPrecision)
    {
        _rwLock.EnterReadLock();
        try
        {
            var layers = ReadAllLayers();
            if (temporalPriors != null)
                InjectTemporalPriors(layers, temporalPriors, temporalPrecision);
            InferCore(layers, input, inferenceSteps, clampTop: false, label: null);
            return (layers[^1].Mu.Clone(), ExtractHiddenStates(layers));
        }
        finally { _rwLock.ExitReadLock(); }
    }

    /// <summary>
    /// Sets (or clears) the <see cref="Model.Layer.PriorMean"/> and
    /// <see cref="Model.Layer.Precision"/> of the top layer.
    /// Pass <c>null</c> to restore a flat (uninformative) prior.
    /// Used by <c>Expanded.MemoryNetwork</c> to inject memory-retrieved priors
    /// before inference without re-creating the network.
    /// </summary>
    public void SetTopLayerPrior(DenseTensor<float>? prior, float precision = 1f)
    {
        _rwLock.EnterWriteLock();
        try
        {
            var top = _store.ReadLayer(LayerCount - 1);
            top.PriorMean = prior?.Clone();
            top.Precision = precision;
            _store.Write(LayerCount - 1, top);
        }
        finally { _rwLock.ExitWriteLock(); }
    }

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose()
    {
        _store.Dispose();
        _rwLock.Dispose();
    }
}
