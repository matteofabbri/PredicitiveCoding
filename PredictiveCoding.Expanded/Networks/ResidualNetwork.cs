using PredictiveCoding.Networks;
using PredictiveCoding.Core;
using PredictiveCoding.Model;

namespace PredictiveCoding.Expanded.Networks;

// ─────────────────────────────────────────────────────────────────────────────
// ResidualNetwork  —  Predictive Coding network with 2-hop skip connections
//
// Adds one skip-connection per adjacent layer pair so that every layer l
// receives top-down predictions from both its direct parent (l+1) AND its
// grandparent (l+2):
//
//   combined_pred_l = f(W_{l+1} μ_{l+1} + b_{l+1})          (adjacent)
//                   + π_skip · g(V_l μ_{l+2} + c_l)          (skip)
//
// The prediction error then minimises the combined mismatch:
//
//   ε_l = Π_l (μ_l − combined_pred_l)
//
// State update for the grandparent layer (μ_{l+2}) gains a skip-gradient term:
//
//   Δμ_{l+2} += η_s · π_skip · V_l^T [g′(pre_skip_l) ⊙ ε_l]
//
// And the skip weights are updated by a matching Hebbian rule:
//
//   ΔV_l  += η_skip · π_skip · [g′(pre_skip_l) ⊙ ε_l] ⊗ μ_{l+2}^T
//   Δc_l  += η_skip · π_skip · [g′(pre_skip_l) ⊙ ε_l]
//
// The result is two complementary gradient highways through every block:
//   • Adjacent path (shallow gradient, fine-grained prediction)
//   • Skip path     (deep gradient, coarse/semantic prediction)
//
// This prevents vanishing prediction errors in deep hierarchies — the exact
// analogue of the ResNet motivation applied to the PC free-energy framework.
//
// Skip weights are persisted in <BasePath>/skip/skip_{tgt}.bin alongside
// the LayerStore on FlushStore() and Dispose() calls.
//
// References:
//   He et al. (2016) "Deep Residual Learning", CVPR.
//   Rao & Ballard (1999); Whittington & Bogacz (2017).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class ResidualNetwork : IDisposable
{
    private readonly LayerStore            _store;
    private readonly ReaderWriterLockSlim  _rwLock = new(LockRecursionPolicy.SupportsRecursion);
    private readonly ResidualNetworkConfig _config;
    private readonly string                _skipDir;

    // Skip weights V[tgt] : [LayerSizes[tgt] × LayerSizes[tgt+2]]
    // Skip biases  c[tgt] : [LayerSizes[tgt]]
    // One entry per target layer tgt ∈ 0..L-3.
    private readonly DenseTensor<float>[] _skipWeights;
    private readonly DenseTensor<float>[] _skipBias;

    // ── Public properties ─────────────────────────────────────────────────────

    public ResidualNetworkConfig Configuration => _config;
    public int LayerCount => _config.BaseConfig.LayerSizes.Length;

    /// <summary>Number of skip connections = LayerCount − 2.</summary>
    public int SkipCount => _skipWeights.Length;

    // ── Construction ─────────────────────────────────────────────────────────

    public ResidualNetwork(ResidualNetworkConfig config, Random rng)
    {
        config.Validate();
        _config = config;

        var   sizes    = config.BaseConfig.LayerSizes;
        int   L        = sizes.Length;
        int   numSkips = L - 2;

        _store   = new LayerStore(config.BaseConfig, rng, config.BaseConfig.LazyPersistence);
        _skipDir = Path.Combine(config.BaseConfig.Path, "skip");
        Directory.CreateDirectory(_skipDir);

        _skipWeights = new DenseTensor<float>[numSkips];
        _skipBias    = new DenseTensor<float>[numSkips];

        for (int tgt = 0; tgt < numSkips; tgt++)
        {
            int   src     = tgt + 2;
            int   tgtSize = sizes[tgt];
            int   srcSize = sizes[src];
            string path   = SkipFilePath(tgt);

            if (File.Exists(path))
                (_skipWeights[tgt], _skipBias[tgt]) = LoadSkipWeights(path, tgtSize, srcSize);
            else
            {
                // V[tgt] shape: [tgtSize × srcSize]  (maps μ_src → prediction at tgt)
                _skipWeights[tgt] = RandomUtils.InitWeights(srcSize, tgtSize,
                                        config.EffectiveSkipActivation, rng);
                _skipBias[tgt]    = RandomUtils.ZeroBias(tgtSize);
            }
        }
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Runs a full inference pass without updating weights.
    /// The inferred state is transient — it is not persisted to the store.
    /// </summary>
    public void Infer(DenseTensor<float> input, int inferenceSteps,
                      bool clampTop = false, DenseTensor<float>? label = null)
    {
        _rwLock.EnterReadLock();
        try
        {
            var layers = ReadAllLayers();
            InferCore(layers, input, inferenceSteps, clampTop, label);
        }
        finally { _rwLock.ExitReadLock(); }
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Learning step: inference with both ends clamped, adjacent + skip weight
    /// updates, then persists to the store.
    /// </summary>
    public void Learn(DenseTensor<float> input, DenseTensor<float> label, int inferenceSteps)
    {
        _rwLock.EnterWriteLock();
        try
        {
            var layers = ReadAllLayers();
            InferCore(layers, input, inferenceSteps, clampTop: true, label: label);

            for (int l = 1; l < layers.Length; l++)
            {
                layers[l].UpdateWeights(layers[l - 1].PredictionError,
                                        _config.BaseConfig.WeightLearningRate);
                _store.Write(l, layers[l]);
            }

            UpdateSkipWeights(layers);
        }
        finally { _rwLock.ExitWriteLock(); }
    }

    /// <summary>
    /// Gated learning step for use inside <see cref="MixtureOfExperts"/>:
    /// scales both adjacent and skip weight learning rates by
    /// <paramref name="gateWeight"/>. Returns the free energy after inference.
    /// When <paramref name="gateWeight"/> == 0 the weight update is skipped.
    /// </summary>
    public float LearnGated(DenseTensor<float> input, DenseTensor<float> label,
                            int inferenceSteps, float gateWeight)
    {
        _rwLock.EnterWriteLock();
        try
        {
            var   layers = ReadAllLayers();
            InferCore(layers, input, inferenceSteps, clampTop: true, label: label);
            float energy = ComputeEnergyCore(layers);

            if (gateWeight > 0f)
            {
                float adjLr  = _config.BaseConfig.WeightLearningRate * gateWeight;
                float skipLr = _config.EffectiveSkipLR * gateWeight;

                for (int l = 1; l < layers.Length; l++)
                {
                    layers[l].UpdateWeights(layers[l - 1].PredictionError, adjLr);
                    _store.Write(l, layers[l]);
                }

                UpdateSkipWeights(layers, skipLr);
            }

            return energy;
        }
        finally { _rwLock.ExitWriteLock(); }
    }

    // ── Energy probe ──────────────────────────────────────────────────────────

    /// <summary>
    /// Runs inference with both ends clamped and returns the free energy without
    /// modifying any weights.  Used for gating in <see cref="MixtureOfExperts"/>.
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

    // ── Prediction ───────────────────────────────────────────────────────────

    /// <summary>
    /// Classifies <paramref name="input"/> via minimum-energy winner-take-all:
    ///   c* = argmin_c  F(x, one-hot(c))
    /// </summary>
    public int Predict(DenseTensor<float> input, int inferenceSteps)
    {
        _rwLock.EnterWriteLock();
        try   { return PredictCore(input, inferenceSteps).cls; }
        finally { _rwLock.ExitWriteLock(); }
    }

    /// <summary>
    /// Classifies <paramref name="input"/> and also returns the winning free energy.
    /// </summary>
    public (int cls, float energy) PredictWithEnergy(DenseTensor<float> input, int inferenceSteps)
    {
        _rwLock.EnterWriteLock();
        try   { return PredictCore(input, inferenceSteps); }
        finally { _rwLock.ExitWriteLock(); }
    }

    /// <summary>
    /// Returns F(x, one-hot(c)) for every class c.
    /// Used by <see cref="EnsembleNetwork"/> for probability averaging.
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

    // ── Core inference loop ───────────────────────────────────────────────────

    private void InferCore(
        Layer[] layers, DenseTensor<float> input, int inferenceSteps,
        bool clampTop, DenseTensor<float>? label)
    {
        ClampLayer(layers, 0, Flatten(input));

        if (clampTop && label != null)
            ClampLayer(layers, layers.Length - 1, FlattenLabel(label));
        else
            layers[^1].IsClamped = false;

        if (_config.BaseConfig.UseFeedforwardGistInit)
            FeedforwardGistInit(layers);

        int   numSkips = _skipWeights.Length;
        var   sizes    = _config.BaseConfig.LayerSizes;
        float actReg   = _config.BaseConfig.ActivityRegularization;
        var   skipAct  = _config.EffectiveSkipActivation;

        // Pre-allocate local skip working buffers (thread-safe; not shared state).
        var skipPreAct = new DenseTensor<float>[numSkips];
        var skipPred   = new DenseTensor<float>[numSkips];
        for (int tgt = 0; tgt < numSkips; tgt++)
        {
            skipPreAct[tgt] = DenseTensor<float>.Zeros(sizes[tgt]);
            skipPred[tgt]   = DenseTensor<float>.Zeros(sizes[tgt]);
        }

        for (int step = 0; step < inferenceSteps; step++)
        {
            // ── 1. Adjacent top-down predictions ─────────────────────────────
            for (int l = layers.Length - 1; l >= 1; l--)
                layers[l].ComputePrediction();

            // ── 2. Skip top-down predictions: layer[tgt+2] → layer[tgt] ─────
            for (int tgt = 0; tgt < numSkips; tgt++)
            {
                int src = tgt + 2;
                // pre_skip = V[tgt] μ_src + c[tgt]
                var pre = TensorUtils.MatVecMul(_skipWeights[tgt], layers[src].Mu);
                TensorUtils.InPlaceAdd(_skipBias[tgt], pre);
                pre.CopyTo(skipPreAct[tgt]);
                // pred_skip = g(pre_skip)
                ActivationFn.Apply(pre, skipAct).CopyTo(skipPred[tgt]);
            }

            // ── 3. Combine: add π_skip · skip_pred into the adjacent prediction
            //    layers[tgt+1].Prediction predicts layer[tgt] (adjacent).
            //    After this step it holds the combined prediction for layer[tgt].
            for (int tgt = 0; tgt < numSkips; tgt++)
                TensorUtils.InPlaceAxpy(_config.SkipPrecision, skipPred[tgt],
                                        layers[tgt + 1].Prediction!);

            // ── 4. Prediction errors using combined predictions ───────────────
            for (int l = 1; l < layers.Length; l++)
                layers[l - 1].SetPredictionError(layers[l].Prediction);
            layers[^1].SetPredictionError(null);

            // ── 5. Standard hidden-state updates (adjacent gradient path) ─────
            for (int l = 1; l < layers.Length; l++)
            {
                if (layers[l].IsClamped) continue;
                layers[l].UpdateState(layers[l - 1].PredictionError,
                                      _config.BaseConfig.StateLearningRate, actReg);
            }

            // ── 6. Skip gradient correction for μ_{tgt+2} ────────────────────
            //    ∇_{μ_src} F_tgt|skip = −π_skip · V[tgt]^T (g′(pre_skip) ⊙ ε_tgt)
            //    μ_src update: += η_s · π_skip · V^T(g′⊙ε)
            for (int tgt = 0; tgt < numSkips; tgt++)
            {
                int src = tgt + 2;
                if (layers[src].IsClamped) continue;

                var gPrime    = ActivationFn.Derivative(skipPreAct[tgt], skipPred[tgt], skipAct);
                var skipDelta = TensorUtils.ElementwiseMul(gPrime, layers[tgt].PredictionError);
                var skipGrad  = TensorUtils.MatVecMulT(_skipWeights[tgt], skipDelta);

                TensorUtils.InPlaceAxpy(
                    _config.BaseConfig.StateLearningRate * _config.SkipPrecision,
                    skipGrad, layers[src].Mu);
            }
        }
    }

    // ── Argmin-energy classification ─────────────────────────────────────────

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

        // Re-run for winner so persisted state reflects the best explanation.
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

    // ── Skip weight update ────────────────────────────────────────────────────

    // Runs one final forward pass through skip connections using the converged
    // layer states, then applies the Hebbian skip weight update:
    //   ΔV[tgt]  += η_skip · π_skip · (g′(pre_skip) ⊙ ε_tgt) ⊗ μ_src^T
    //   Δc[tgt]  += η_skip · π_skip · (g′(pre_skip) ⊙ ε_tgt)
    private void UpdateSkipWeights(Layer[] layers, float? overrideLr = null)
    {
        float eta  = (overrideLr ?? _config.EffectiveSkipLR) * _config.SkipPrecision;
        var   act  = _config.EffectiveSkipActivation;

        for (int tgt = 0; tgt < _skipWeights.Length; tgt++)
        {
            int src = tgt + 2;

            var pre   = TensorUtils.MatVecMul(_skipWeights[tgt], layers[src].Mu);
            TensorUtils.InPlaceAdd(_skipBias[tgt], pre);
            var pred  = ActivationFn.Apply(pre, act);

            var gPrime    = ActivationFn.Derivative(pre, pred, act);
            var skipDelta = TensorUtils.ElementwiseMul(gPrime, layers[tgt].PredictionError);

            TensorUtils.InPlaceAxpy(eta, TensorUtils.OuterProduct(skipDelta, layers[src].Mu),
                                    _skipWeights[tgt]);
            TensorUtils.InPlaceAxpy(eta, skipDelta, _skipBias[tgt]);
        }
    }

    // ── Energy ───────────────────────────────────────────────────────────────

    /// <summary>Global free energy F = ½ Σ_{l=0}^{L-2} ||ε_l||² loaded from store.</summary>
    public float ComputeEnergy()
    {
        _rwLock.EnterReadLock();
        try
        {
            float e = 0f;
            for (int l = 0; l < LayerCount - 1; l++)
                e += _store.ReadLayer(l).LocalEnergy();
            return e;
        }
        finally { _rwLock.ExitReadLock(); }
    }

    private static float ComputeEnergyCore(Layer[] layers)
    {
        float e = 0f;
        for (int l = 0; l < layers.Length - 1; l++)
            e += layers[l].LocalEnergy();
        return e;
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    /// <summary>Average L2 norm of adjacent generative weights.</summary>
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

    /// <summary>
    /// Per-skip L2 weight norms (length = SkipCount).
    /// Useful for monitoring whether skip connections are being used.
    /// </summary>
    public float[] SkipWeightNorms()
    {
        var norms = new float[_skipWeights.Length];
        for (int i = 0; i < _skipWeights.Length; i++)
            norms[i] = MathF.Sqrt(TensorUtils.SumSquares(_skipWeights[i]));
        return norms;
    }

    public void PrintArchitecture()
    {
        Console.WriteLine("═══ ResidualNetwork Architecture ═══");
        var sizes = _config.BaseConfig.LayerSizes;
        for (int l = 0; l < LayerCount; l++)
        {
            string adjW = l > 0 ? $"  W[{sizes[l-1]}←{sizes[l]}]" : "";
            Console.WriteLine($"  Layer[{l}]: μ=[{sizes[l]}]{adjW}");
            if (l >= 2)
                Console.WriteLine($"      └─ skip V[{sizes[l-2]}←{sizes[l]}] (from layer {l} to {l-2})");
        }
        Console.WriteLine($"  π_skip={_config.SkipPrecision:F3}  " +
                          $"η_skip={_config.EffectiveSkipLR:G3}  " +
                          $"skip_act={_config.EffectiveSkipActivation}");
        Console.WriteLine($"  η_state={_config.BaseConfig.StateLearningRate:F4}  " +
                          $"η_weight={_config.BaseConfig.WeightLearningRate:F4}");
        Console.WriteLine("═════════════════════════════════════");
    }

    // ── Persistence ──────────────────────────────────────────────────────────

    /// <summary>
    /// Flushes the adjacent-weight store and saves all skip weight matrices.
    /// Required when <see cref="NetworkConfig.LazyPersistence"/> is true, or
    /// to checkpoint skip weights after training.
    /// </summary>
    public void FlushStore()
    {
        _rwLock.EnterWriteLock();
        try
        {
            _store.Flush();
            SaveAllSkipWeights();
        }
        finally { _rwLock.ExitWriteLock(); }
    }

    private void SaveAllSkipWeights()
    {
        for (int tgt = 0; tgt < _skipWeights.Length; tgt++)
            SaveSkipWeights(tgt);
    }

    private void SaveSkipWeights(int tgt)
    {
        var   W    = _skipWeights[tgt];
        var   b    = _skipBias[tgt];
        int   rows = W.Dimensions[0];
        int   cols = W.Dimensions[1];

        using var fs = new FileStream(SkipFilePath(tgt), FileMode.Create, FileAccess.Write);
        using var bw = new BinaryWriter(fs);
        bw.Write(rows);
        bw.Write(cols);
        for (int i = 0; i < rows * cols; i++) bw.Write(W.Buffer[i]);
        for (int i = 0; i < rows;       i++) bw.Write(b.Buffer[i]);
    }

    private static (DenseTensor<float> W, DenseTensor<float> b) LoadSkipWeights(
        string path, int expectedRows, int expectedCols)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var br = new BinaryReader(fs);

        int rows = br.ReadInt32();
        int cols = br.ReadInt32();

        if (rows != expectedRows || cols != expectedCols)
            throw new InvalidDataException(
                $"Skip weight file {path} has shape [{rows}×{cols}] " +
                $"but the current config expects [{expectedRows}×{expectedCols}]. " +
                "Delete the file to reinitialise.");

        var W = new DenseTensor<float>(rows, cols);
        var b = DenseTensor<float>.Zeros(rows);
        for (int i = 0; i < rows * cols; i++) W.Buffer[i] = br.ReadSingle();
        for (int i = 0; i < rows;        i++) b.Buffer[i] = br.ReadSingle();
        return (W, b);
    }

    private string SkipFilePath(int tgt) =>
        Path.Combine(_skipDir, $"skip_{tgt}.bin");

    // ── Private helpers ───────────────────────────────────────────────────────

    private Layer[] ReadAllLayers()
    {
        var layers = new Layer[LayerCount];
        for (int l = 0; l < LayerCount; l++)
            layers[l] = _store.ReadLayer(l);
        return layers;
    }

    private static void ClampLayer(Layer[] layers, int index, DenseTensor<float> value)
    {
        value.CopyTo(layers[index].Mu);
        layers[index].IsClamped = true;
    }

    private static void FeedforwardGistInit(Layer[] layers)
    {
        for (int l = layers.Length - 1; l >= 1; l--)
        {
            layers[l].ComputePrediction();
            if (l > 1 && !layers[l - 1].IsClamped)
                layers[l].Prediction!.CopyTo(layers[l - 1].Mu);
        }
    }

    private DenseTensor<float> Flatten(DenseTensor<float> input)
    {
        int expected = _config.BaseConfig.LayerSizes[0];
        if (input.Length != expected)
            throw new ArgumentException(
                $"Input has {input.Length} elements but the network expects {expected} " +
                $"(LayerSizes[0]).");
        return input.Rank == 1 ? input : TensorUtils.Reshape(input, expected);
    }

    private DenseTensor<float> FlattenLabel(DenseTensor<float> label)
    {
        int expected = _config.BaseConfig.LayerSizes[^1];
        if (label.Length != expected)
            throw new ArgumentException(
                $"Label has {label.Length} elements but the network expects {expected} " +
                $"(LayerSizes[^1]).");
        return label.Rank == 1 ? label : TensorUtils.Reshape(label, expected);
    }

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose()
    {
        _rwLock.EnterWriteLock();
        try   { SaveAllSkipWeights(); }
        finally { _rwLock.ExitWriteLock(); }

        _store.Dispose();
        _rwLock.Dispose();
    }
}
