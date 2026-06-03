using PredictiveCoding.Networks;
using PredictiveCoding.Core;
using PredictiveCoding.Model;

namespace PredictiveCoding.Expanded.Networks;

// ─────────────────────────────────────────────────────────────────────────────
// AttentiveNetwork  —  PC network with per-unit learnable attention gates
//
// Each layer l ∈ {0..L-2} carries a learned attention vector
//   α_l = σ(s_l)  ∈ (0, 1)^{size_l}
// where s_l ∈ ℝ^{size_l} is a learnable logit (initialised to 0 → α = 0.5).
//
// During inference, before the prediction error ε_{l-1} is used in the
// state and weight updates of layer l, it is gated by α_{l-1}:
//
//   ε_{l-1}^att = α_{l-1} ⊙ ε_{l-1}
//
// This controls how strongly each unit's prediction error drives learning
// in the layer above, selectively amplifying or suppressing individual
// dimensions of the hierarchy's feedback signal.
//
// The attention logits are updated once per learning step (not per inference
// step) by ascending the gradient of a saliency objective with L2 regularisation:
//
//   Δs_l[i] = η_att · (|ε_l[i]| · α_l[i] · (1 − α_l[i]) − λ_att · s_l[i])
//
// Interpretation:
//   • Gradient term |ε| · α(1−α): pushes logit up when the unit has a large
//     prediction error AND is near α = 0.5 (maximally responsive region of
//     the sigmoid).  The attention grows toward units that are consistently
//     hard to predict — the most informative dimensions.
//   • Regulariser −λ · s: pulls the logit back toward 0 (α = 0.5, neutral)
//     if the error signal weakens, preventing runaway specialisation.
//
// At the fixed point α_l[i]* encodes the long-run prediction difficulty of
// unit i in layer l: reliably predicted units settle near 0.5, informative
// (hard-to-predict) units settle near 1.  Visualising α after training
// reveals which dimensions the hierarchy "attends to."
//
// Attention logits are persisted in <BasePath>/attention/attn_{l}.bin.
//
// References:
//   Rao & Ballard (1999); Friston (2005) (precision weighting in PC)
//   Vaswani et al. (2017) (attention mechanism in transformer)
// ─────────────────────────────────────────────────────────────────────────────

public sealed class AttentiveNetwork : IDisposable
{
    private readonly LayerStore              _store;
    private readonly ReaderWriterLockSlim    _rwLock = new(LockRecursionPolicy.SupportsRecursion);
    private readonly AttentiveNetworkConfig  _config;
    private readonly string                  _attDir;

    // Attention logits s_l for l = 0..L-2 (all layers except the top output).
    // α_l = σ(s_l).  Initialised to 0 → α = 0.5 (neutral).
    private readonly DenseTensor<float>[]    _attLogits;

    // ── Public properties ─────────────────────────────────────────────────────

    public AttentiveNetworkConfig Configuration => _config;
    public int LayerCount  => _config.BaseConfig.LayerSizes.Length;

    /// <summary>Number of attention vectors = LayerCount − 1 (all but the top layer).</summary>
    public int AttentionCount => _attLogits.Length;

    // ── Construction ─────────────────────────────────────────────────────────

    public AttentiveNetwork(AttentiveNetworkConfig config, Random rng)
    {
        config.Validate();
        _config = config;

        var  sizes    = config.BaseConfig.LayerSizes;
        int  L        = sizes.Length;
        int  numAtt   = L - 1;   // layers 0..L-2

        _store  = new LayerStore(config.BaseConfig, rng, config.BaseConfig.LazyPersistence);
        _attDir = Path.Combine(config.BaseConfig.Path, "attention");
        Directory.CreateDirectory(_attDir);

        _attLogits = new DenseTensor<float>[numAtt];
        for (int l = 0; l < numAtt; l++)
        {
            string path = AttLogitPath(l);
            if (File.Exists(path))
                _attLogits[l] = LoadLogits(path, sizes[l]);
            else
            {
                // Zero-initialised → α = σ(0) = 0.5, neutral attention at the start.
                _attLogits[l] = DenseTensor<float>.Zeros(sizes[l]);
            }
        }
    }

    // ── Inference ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Runs a full inference pass using the current attention profile.
    /// Transient: weights and attention logits are not modified.
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
    /// One learning step: inference with both ends clamped, Hebbian weight update
    /// using attended errors, then attention logit update.
    /// </summary>
    public void Learn(DenseTensor<float> input, DenseTensor<float> label, int inferenceSteps)
    {
        _rwLock.EnterWriteLock();
        try
        {
            var layers = ReadAllLayers();
            InferCore(layers, input, inferenceSteps, clampTop: true, label: label);
            UpdateWeights(layers, _config.BaseConfig.WeightLearningRate);
            UpdateAttentionLogits(layers);
        }
        finally { _rwLock.ExitWriteLock(); }
    }

    /// <summary>
    /// Gated learning step for use in <see cref="MixtureOfExperts"/>: scales
    /// the weight learning rate by <paramref name="gateWeight"/>.
    /// Returns the free energy after inference.
    /// When <paramref name="gateWeight"/> == 0 no weights or logits are updated.
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
                UpdateWeights(layers, _config.BaseConfig.WeightLearningRate * gateWeight);
                UpdateAttentionLogits(layers);
            }

            return energy;
        }
        finally { _rwLock.ExitWriteLock(); }
    }

    // ── Energy probe ──────────────────────────────────────────────────────────

    /// <summary>
    /// Runs inference with both ends clamped and returns the free energy without
    /// modifying weights or logits.
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
    /// Classifies <paramref name="input"/> via minimum-energy winner-take-all.
    /// </summary>
    public int Predict(DenseTensor<float> input, int inferenceSteps)
    {
        _rwLock.EnterWriteLock();
        try   { return PredictCore(input, inferenceSteps).cls; }
        finally { _rwLock.ExitWriteLock(); }
    }

    /// <summary>Classifies and returns the winning free energy.</summary>
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

        float actReg = _config.BaseConfig.ActivityRegularization;
        float etaS   = _config.BaseConfig.StateLearningRate;

        // Local attended-error buffers: one per attention layer (l = 0..L-2).
        // Allocated once per InferCore call so concurrent read-locked callers
        // (ClassEnergies, InferEnergy) never share buffers.
        var attErrors = AllocAttendedErrors();

        for (int step = 0; step < inferenceSteps; step++)
        {
            // ── Top-down: each layer predicts the one below ───────────────────
            for (int l = layers.Length - 1; l >= 1; l--)
                layers[l].ComputePrediction();

            // ── Bottom-up: ε_{l-1} = μ_{l-1} − pred_l ───────────────────────
            for (int l = 1; l < layers.Length; l++)
                layers[l - 1].SetPredictionError(layers[l].Prediction);
            layers[^1].SetPredictionError(null);

            // ── Attend: ε_l^att = α_l ⊙ ε_l  for l = 0..L-2 ─────────────────
            for (int l = 0; l < _attLogits.Length; l++)
                ApplyAttention(layers[l].PredictionError, l, attErrors[l]);

            // ── State update: μ_l uses attended error from the layer below ────
            for (int l = 1; l < layers.Length; l++)
            {
                if (layers[l].IsClamped) continue;
                layers[l].UpdateState(attErrors[l - 1], etaS, actReg);
            }
        }
    }

    // ── Weight and attention updates ─────────────────────────────────────────

    private void UpdateWeights(Layer[] layers, float etaW)
    {
        // Re-compute attended errors at the converged state for the weight update.
        // This is a single forward pass (no inference step) — cheap.
        var attErrors = AllocAttendedErrors();
        for (int l = 0; l < _attLogits.Length; l++)
            ApplyAttention(layers[l].PredictionError, l, attErrors[l]);

        for (int l = 1; l < layers.Length; l++)
        {
            layers[l].UpdateWeights(attErrors[l - 1], etaW);
            _store.Write(l, layers[l]);
        }
    }

    /// <summary>
    /// Updates each attention logit s_l after one learning step:
    ///   Δs_l[i] = η_att · (|ε_l[i]| · α_l[i] · (1 − α_l[i]) − λ_att · s_l[i])
    ///
    /// The first term pushes the logit up for units with large prediction error
    /// (attend more to informative dimensions).  The second term provides L2
    /// regularisation that anchors the logit near 0 (α = 0.5, neutral attention)
    /// whenever the error signal is weak.
    /// </summary>
    private void UpdateAttentionLogits(Layer[] layers)
    {
        float eta = _config.AttentionLearningRate;
        float lam = _config.AttentionRegularization;

        for (int l = 0; l < _attLogits.Length; l++)
        {
            var s   = _attLogits[l];
            var eps = layers[l].PredictionError;

            for (int i = 0; i < s.Length; i++)
            {
                float ai   = Sigmoid(s[i]);
                float grad = MathF.Abs(eps[i]) * ai * (1f - ai) - lam * s[i];
                s[i] += eta * grad;
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

        // Re-run for winner so the persisted state reflects the best explanation.
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

    // ── Attention helpers ─────────────────────────────────────────────────────

    // Allocates a fresh set of attended-error buffers (one per attention layer).
    // Called per InferCore to avoid sharing buffers across concurrent callers.
    private DenseTensor<float>[] AllocAttendedErrors()
    {
        var   sizes  = _config.BaseConfig.LayerSizes;
        var   result = new DenseTensor<float>[_attLogits.Length];
        for (int l = 0; l < result.Length; l++)
            result[l] = DenseTensor<float>.Zeros(sizes[l]);
        return result;
    }

    // Writes α_l ⊙ error into dest in-place.
    private void ApplyAttention(DenseTensor<float> error, int l, DenseTensor<float> dest)
    {
        var s = _attLogits[l];
        for (int i = 0; i < error.Length; i++)
            dest.Buffer[i] = Sigmoid(s[i]) * error[i];
    }

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    // ── Diagnostics ──────────────────────────────────────────────────────────

    /// <summary>
    /// Returns a copy of the current attention weights α_l = σ(s_l) for layer
    /// <paramref name="layerIndex"/>.  Values in (0, 1): higher means the unit
    /// receives more weight in the feedback signal to the layer above.
    /// </summary>
    public float[] AttentionWeights(int layerIndex)
    {
        if ((uint)layerIndex >= (uint)_attLogits.Length)
            throw new ArgumentOutOfRangeException(nameof(layerIndex),
                $"Layer index must be in [0, {_attLogits.Length - 1}].");
        _rwLock.EnterReadLock();
        try
        {
            var s      = _attLogits[layerIndex];
            var result = new float[s.Length];
            for (int i = 0; i < s.Length; i++)
                result[i] = Sigmoid(s[i]);
            return result;
        }
        finally { _rwLock.ExitReadLock(); }
    }

    /// <summary>Mean attention weight across all units in layer <paramref name="layerIndex"/>.</summary>
    public float MeanAttention(int layerIndex)
    {
        var weights = AttentionWeights(layerIndex);
        float sum = 0f;
        foreach (float w in weights) sum += w;
        return sum / weights.Length;
    }

    /// <summary>
    /// Binary entropy of the attention distribution at each layer.
    /// Higher entropy → more uniform (diffuse) attention.
    /// Lower entropy → more peaked (selective) attention.
    /// </summary>
    public float AttentionEntropy(int layerIndex)
    {
        var weights = AttentionWeights(layerIndex);
        float h = 0f;
        foreach (float a in weights)
        {
            float b = MathF.Max(1e-7f, 1f - a);
            float c = MathF.Max(1e-7f, a);
            h -= a * MathF.Log(c) + (1f - a) * MathF.Log(b);
        }
        return h / weights.Length;   // normalised by layer size
    }

    /// <summary>Average weight norm of adjacent generative weights.</summary>
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

    public void PrintAttentionProfile()
    {
        _rwLock.EnterReadLock();
        try
        {
            Console.WriteLine("═══ Attention Profile ═══");
            for (int l = 0; l < _attLogits.Length; l++)
            {
                var alpha    = AttentionWeights(l);
                float mean   = alpha.Average();
                float maxVal = alpha.Max();
                float minVal = alpha.Min();
                Console.WriteLine(
                    $"  Layer[{l}]: μ_α={mean:F3}  max={maxVal:F3}  min={minVal:F3}  " +
                    $"H={AttentionEntropy(l):F3}");
            }
            Console.WriteLine("═════════════════════════");
        }
        finally { _rwLock.ExitReadLock(); }
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

    // ── Persistence ──────────────────────────────────────────────────────────

    /// <summary>
    /// Flushes the adjacent-weight store and saves all attention logits to disk.
    /// Required when <see cref="NetworkConfig.LazyPersistence"/> is true, or
    /// to checkpoint the attention profile after training.
    /// </summary>
    public void FlushStore()
    {
        _rwLock.EnterWriteLock();
        try
        {
            _store.Flush();
            SaveAllLogits();
        }
        finally { _rwLock.ExitWriteLock(); }
    }

    private void SaveAllLogits()
    {
        for (int l = 0; l < _attLogits.Length; l++)
        {
            using var fs = new FileStream(AttLogitPath(l), FileMode.Create, FileAccess.Write);
            using var bw = new BinaryWriter(fs);
            bw.Write(_attLogits[l].Length);
            foreach (float v in _attLogits[l].Buffer) bw.Write(v);
        }
    }

    private static DenseTensor<float> LoadLogits(string path, int expectedSize)
    {
        using var fs = new FileStream(path, FileMode.Open, FileAccess.Read);
        using var br = new BinaryReader(fs);
        int size = br.ReadInt32();
        if (size != expectedSize)
            throw new InvalidDataException(
                $"Attention logit file {path} has {size} units " +
                $"but the current config expects {expectedSize}. " +
                "Delete the file to reinitialise.");
        var t = DenseTensor<float>.Zeros(size);
        for (int i = 0; i < size; i++) t.Buffer[i] = br.ReadSingle();
        return t;
    }

    private string AttLogitPath(int l) =>
        Path.Combine(_attDir, $"attn_{l}.bin");

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
                $"Input has {input.Length} elements but the network expects {expected}.");
        return input.Rank == 1 ? input : TensorUtils.Reshape(input, expected);
    }

    private DenseTensor<float> FlattenLabel(DenseTensor<float> label)
    {
        int expected = _config.BaseConfig.LayerSizes[^1];
        if (label.Length != expected)
            throw new ArgumentException(
                $"Label has {label.Length} elements but the network expects {expected}.");
        return label.Rank == 1 ? label : TensorUtils.Reshape(label, expected);
    }

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose()
    {
        _rwLock.EnterWriteLock();
        try   { SaveAllLogits(); }
        finally { _rwLock.ExitWriteLock(); }

        _store.Dispose();
        _rwLock.Dispose();
    }
}
