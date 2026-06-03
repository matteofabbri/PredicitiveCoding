using System.Runtime.CompilerServices;
using PredictiveCoding.Core;
using PredictiveCoding.Networks;

[assembly: InternalsVisibleTo("PredicitiveCoding.UnitTests")]
[assembly: InternalsVisibleTo("PredictiveCoding.Expanded")]

namespace PredictiveCoding.Model;

// ─────────────────────────────────────────────────────────────────────────────
// PCLayer  —  Single layer in a hierarchical predictive coding network
//
// State variables:
//   μ_l      : latent state                    [Size]
//   W_l      : generative weights              [LowerSize × Size]  (null if input layer)
//   b_l      : generative bias                 [LowerSize]          (null if input layer)
//   pre_l    : pre-activation W_l μ_l + b_l   [LowerSize]
//   pred_l   : top-down prediction f(pre_l)   [LowerSize]
//   ε_l      : precision-weighted error        [Size]
//              ε_l = Π_l (μ_l − pred_{l+1→l})     (hidden/input layers)
//              ε_l = Π_l (μ_l − v_p_l)             (top layer with informative prior)
//              ε_l = 0                              (top layer, flat prior)
//   Precision  (Π_l): isotropic precision scalar (default 1)
//   PriorMean  (v_p): prior mean for the top layer (null = flat prior)
//
// Free-energy contribution (Bogacz 2017, eq. 1 / eq. 6):
//   F_l = ½ Π_l ||μ_l − pred_{l+1→l}||²
//       = ½ ||ε_l||² / Π_l          (since ε_l = Π_l · residual_l)
//
// References:
//   Bogacz (2017) J. Math. Psych. 76:198-211
//   Rao & Ballard (1999); Whittington & Bogacz (2017)
// ─────────────────────────────────────────────────────────────────────────────

public sealed class Layer
{
    // ── Structure ────────────────────────────────────────────────────────────
    public int Index { get; init; }   // index in network (0 = sensory)

    public int Size { get; init; }   // dimensionality of this layer
    public int LowerSize { get; init; }   // dimensionality of the layer below (0 if input)

    // ── Hyperparameters ───────────────────────────────────────────────────────
    public float StateLearningRate { get; internal set; }

    public float WeightLearningRate { get; internal set; }
    public ActivationType ActivationType { get; init; }

    // ── Bogacz (2017) precision and prior (eq. 6) ─────────────────────────────

    /// <summary>
    /// Π_l : isotropic precision (inverse variance) for this layer's prediction error.
    /// Scales ε_l = Π_l · (μ_l − prediction).  Default = 1 (unit precision).
    ///
    /// Setting Precision on the sensory layer (Π_x) vs. the top layer (Π_v)
    /// controls the relative trust in data vs. prior (Bogacz eq. 11):
    ///   v* = (θ^T Π_x θ + Π_v)^{-1} θ^T Π_x x
    ///
    /// Ignored when DiagonalPrecision is non-null.
    /// </summary>
    public float Precision { get; internal set; } = 1f;

    /// <summary>
    /// Π_l[i] : per-unit (diagonal) precision vector  [Size].
    /// Implements the diagonal form of the Huang &amp; Rao (2011) covariance matrix
    /// Σ_l, where each unit i has its own noise variance σ²_l[i] = 1/Π_l[i].
    ///
    /// When non-null, overrides the scalar <see cref="Precision"/>:
    ///   ε_l[i] = Π_l[i] · (μ_l[i] − prediction_l[i])
    ///
    /// Diagonal precision update (gradient of free energy per unit, Huang &amp; Rao eq. 3):
    ///   ΔΠ_l[i] = η · (1/Π_l[i] − residual_l[i]²) / 2
    /// At the MLE fixed point: Π_l[i]* = 1 / residual_l[i]²
    ///
    /// Null (default) → fall back to scalar <see cref="Precision"/>.
    /// </summary>
    public DenseTensor<float>? DiagonalPrecision { get; internal set; }

    /// <summary>
    /// λ_l : L2 weight decay coefficient (Huang &amp; Rao 2011, regularisation term λ||W||²).
    /// Applied during weight updates: ΔW -= λ · W.
    /// Default = 0 (no regularisation).
    /// </summary>
    public float WeightDecay { get; internal set; } = 0f;

    /// <summary>
    /// μ_l^{t-1} : hidden state from the previous timestep, used as a temporal
    /// prior during inference (Friston 2008 dynamic causal modelling).
    /// When non-null, adds a quadratic penalty toward the previous state:
    ///   grad_l += Π_t · (μ_l − μ_l^{t-1})
    /// Managed externally by <see cref="TemporalNetwork"/>; not persisted in
    /// <see cref="LayerStore"/> (it is transient working memory, not a weight).
    /// </summary>
    public DenseTensor<float>? TemporalPrior { get; internal set; }

    /// <summary>
    /// Π_t : temporal precision — strength of the continuity pull toward the
    /// previous state <see cref="TemporalPrior"/>.  0 = disabled (default).
    /// </summary>
    public float TemporalPrecision { get; internal set; } = 0f;

    /// <summary>
    /// v_p : prior mean for the top layer (Bogacz eq. 6: ε_v = Π_v (v − v_p)).
    /// Null (default) → flat prior (ε_top = 0, MLE inference).
    /// Set on the top-most layer to obtain proper Bayesian inference.
    /// </summary>
    public DenseTensor<float>? PriorMean { get; internal set; }

    // ── Flag ─────────────────────────────────────────────────────────────────
    public bool IsClamped { get; internal set; }

    // ── Latent state ─────────────────────────────────────────────────────────
    /// <summary>μ_l : current latent state  [Size]</summary>
    public DenseTensor<float> Mu { get; internal set; } = null!;

    // ── Generative model (null for the input layer) ───────────────────────────
    /// <summary>W_l : generative weights  [LowerSize × Size]</summary>
    public DenseTensor<float>? GenerativeWeights { get; internal set; }

    /// <summary>b_l : generative bias  [LowerSize]</summary>
    public DenseTensor<float>? Bias { get; internal set; }

    // ── Computed quantities ────────────────────────────────────────────────────
    /// <summary>pre_l = W_l μ_l + b_l  [LowerSize]</summary>
    public DenseTensor<float>? PreActivation { get; internal set; }

    /// <summary>pred_l = f(pre_l)  [LowerSize]</summary>
    public DenseTensor<float>? Prediction { get; internal set; }

    /// <summary>
    /// ε_l = Π_l (μ_l − prediction received from layer l+1)  [Size]
    /// For the top layer with a prior: ε_l = Π_l (μ_l − v_p).
    /// </summary>
    public DenseTensor<float> PredictionError { get; internal set; } = null!;

    // ── Initialisation ───────────────────────────────────────────────────────

    /// <summary>
    /// Allocates all tensors to zero without random initialisation.
    /// Used when deserializing from <see cref="LayerStore"/> — the store
    /// overwrites the buffers immediately after allocation.
    /// </summary>
    internal void AllocateTensors(bool hasDiagonalPrecision = false, bool hasPriorMean = false)
    {
        Mu = DenseTensor<float>.Zeros(Size);
        PredictionError = DenseTensor<float>.Zeros(Size);
        if (hasDiagonalPrecision) DiagonalPrecision = DenseTensor<float>.Zeros(Size);
        if (hasPriorMean) PriorMean = DenseTensor<float>.Zeros(Size);
        if (LowerSize == 0) return;
        GenerativeWeights = new DenseTensor<float>(LowerSize, Size);
        Bias = DenseTensor<float>.Zeros(LowerSize);
        PreActivation = DenseTensor<float>.Zeros(LowerSize);
        Prediction = DenseTensor<float>.Zeros(LowerSize);
    }

    public void Initialize(Random rng)
    {
        Mu = RandomUtils.SampleNormal([Size], 0f, 0.01f, rng);
        PredictionError = DenseTensor<float>.Zeros(Size);

        if (LowerSize == 0) return;

        GenerativeWeights = RandomUtils.InitWeights(Size, LowerSize, ActivationType, rng);
        Bias = RandomUtils.ZeroBias(LowerSize);
        PreActivation = DenseTensor<float>.Zeros(LowerSize);
        Prediction = DenseTensor<float>.Zeros(LowerSize);
    }

    /// <summary>
    /// Initialises <see cref="DiagonalPrecision"/> to all-ones (unit precision per unit),
    /// enabling per-unit precision learning (Huang &amp; Rao 2011).
    /// Call after <see cref="Initialize"/>.
    /// </summary>
    public void InitDiagonalPrecision()
    {
        DiagonalPrecision = DenseTensor<float>.Zeros(Size);
        DiagonalPrecision.Fill(1f);
    }

    // ── Forward: top-down prediction ─────────────────────────────────────────

    /// <summary>pred_l = f(W_l μ_l + b_l)   (Rao &amp; Ballard eq. 2)</summary>
    public void ComputePrediction()
    {
        if (GenerativeWeights == null) return;

        var pre = TensorUtils.MatVecMul(GenerativeWeights, Mu);
        TensorUtils.InPlaceAdd(Bias!, pre);
        ActivationFn.Apply(pre, ActivationType).CopyTo(Prediction!);
    }

    // ── Prediction error (Bogacz eq. 6) ───────────────────────────────────────

    /// <summary>
    /// Computes the precision-weighted prediction error (Bogacz 2017, eq. 6):
    ///   ε_l = Π_l · (μ_l − predFromAbove)        hidden / input layers
    ///   ε_l = Π_l · (μ_l − v_p)                  top layer with PriorMean set
    ///   ε_l = 0                                    top layer, flat prior (PriorMean null)
    ///
    /// Precision > 1 increases the weight of this layer's error in the global
    /// free energy and in the state/weight update gradients.
    /// </summary>
    public void SetPredictionError(DenseTensor<float>? predFromAbove)
    {
        DenseTensor<float> residual;

        if (predFromAbove == null)
        {
            // Top layer: ε = Π (μ − v_p), or zero for flat prior
            if (PriorMean == null) { PredictionError.Zero(); return; }
            ShapeCheck.SameLength(Mu, PriorMean);
            residual = TensorUtils.ElementwiseSub(Mu, PriorMean);
        }
        else
        {
            ShapeCheck.SameLength(Mu, predFromAbove);
            residual = TensorUtils.ElementwiseSub(Mu, predFromAbove);
        }

        // Scale by precision: ε_l = Π_l · residual_l
        if (DiagonalPrecision != null)
            TensorUtils.ElementwiseMul(residual, DiagonalPrecision).CopyTo(PredictionError);
        else if (Precision == 1f)
            residual.CopyTo(PredictionError);
        else
            TensorUtils.ScalarMul(residual, Precision).CopyTo(PredictionError);
    }

    // ── State update (inference step) ────────────────────────────────────────

    /// <summary>
    /// μ_l ← μ_l − η_s (ε_l − W_l^T [f'(pre_l) ⊙ ε_{l-1}] + λ_act · sign(μ_l))
    ///
    /// Because ε already carries the precision weighting (eq. 6), this gradient
    /// automatically implements the full Bogacz eq. 5:
    ///   dv/dt = g'(v)^T ε_x − ε_v
    /// (Rao &amp; Ballard eq. 2.3; Whittington &amp; Bogacz eq. 14)
    ///
    /// The optional <paramref name="activityReg"/> term adds an L1 sparsity
    /// penalty on μ_l, encouraging sparse hidden representations.
    /// </summary>
    public void UpdateState(DenseTensor<float>? lowerLayerError, float etaState, float activityReg = 0f)
    {
        if (IsClamped || GenerativeWeights == null) return;

        DenseTensor<float> grad;
        if (lowerLayerError != null)
        {
            var fPrime = ActivationFn.Derivative(PreActivation!, Prediction, ActivationType);
            var delta = TensorUtils.ElementwiseMul(fPrime, lowerLayerError);
            var wTDelta = TensorUtils.MatVecMulT(GenerativeWeights, delta);
            grad = TensorUtils.ElementwiseSub(PredictionError, wTDelta);
        }
        else
        {
            grad = PredictionError.Clone();
        }

        // L1 activity regularisation: grad += λ_act · sign(μ_l)
        if (activityReg > 0f)
        {
            for (int i = 0; i < Size; i++)
                grad.Buffer[i] += activityReg * MathF.Sign(Mu[i]);
        }

        // Temporal prior: grad += Π_t · (μ_l − μ_l^{t-1})
        // Pulls the current state toward the previous timestep's state,
        // implementing temporal continuity (Friston 2008).
        if (TemporalPrecision > 0f && TemporalPrior != null)
        {
            for (int i = 0; i < Size; i++)
                grad.Buffer[i] += TemporalPrecision * (Mu[i] - TemporalPrior[i]);
        }

        TensorUtils.InPlaceAxpy(-etaState, grad, Mu);
    }

    // ── Weight update (learning step) ─────────────────────────────────────────

    /// <summary>
    /// W_l ← W_l + η_W (f'(pre_l) ⊙ ε_{l-1}) ⊗ μ_l^T
    ///
    /// ε_{l-1} is already precision-weighted, so this implements the full
    /// Bogacz eq. 7:  dθ/dt = ε_x · v^T
    /// (Whittington &amp; Bogacz eq. 20; Hebbian rule weighted by prediction error)
    /// </summary>
    public void UpdateWeights(DenseTensor<float>? lowerLayerError, float etaWeight)
    {
        if (GenerativeWeights == null || lowerLayerError == null) return;

        var fPrime = ActivationFn.Derivative(PreActivation!, Prediction, ActivationType);
        var delta = TensorUtils.ElementwiseMul(fPrime, lowerLayerError);
        var deltaW = TensorUtils.OuterProduct(delta, Mu);
        TensorUtils.InPlaceAxpy(etaWeight, deltaW, GenerativeWeights);
        TensorUtils.InPlaceAxpy(etaWeight, delta, Bias!);

        // L2 regularisation: ΔW -= λ · W  (Huang & Rao 2011, loss term λ||W||²)
        if (WeightDecay > 0f)
            TensorUtils.InPlaceAxpy(-etaWeight * WeightDecay, GenerativeWeights, GenerativeWeights);
    }

    // ── Precision update (Bogacz 2017, section 2.4.2) ─────────────────────────

    /// <summary>
    /// Gradient descent on the free energy w.r.t. the layer precision Π_l.
    ///
    /// Free-energy contribution including the log-normaliser:
    ///   F_l = ½ Π_l ||residual_l||² − ½ n_l log Π_l
    ///
    /// Gradient:
    ///   ∂F_l/∂Π_l = ½ ||ε_l||²/Π_l² − ½ n_l/Π_l
    ///
    /// Update:
    ///   Π_l ← max(ε, Π_l − η_Π · ∂F_l/∂Π_l)
    ///
    /// At the MLE fixed point: Π_l* = n_l / ||residual_l||²
    /// (Friston 2005; Bogacz 2017 section 2.4.2)
    /// </summary>
    public void UpdatePrecision(float eta)
    {
        // residual = ε / Π  →  ||residual||² = ||ε||² / Π²
        float sumSqResidual = TensorUtils.SumSquares(PredictionError) / (Precision * Precision);
        float grad = 0.5f * sumSqResidual - 0.5f * Size / Precision;
        Precision = MathF.Max(1e-6f, Precision - eta * grad);
    }

    // ── Diagonal precision update (Huang & Rao 2011) ──────────────────────────

    /// <summary>
    /// Per-unit gradient descent on the free energy w.r.t. each diagonal precision
    /// element Π_l[i]  (Huang &amp; Rao 2011, diagonal form of Σ update).
    ///
    /// Free-energy contribution per unit:
    ///   F_l[i] = ½ Π_l[i] residual_l[i]² − ½ log Π_l[i]
    ///
    /// Gradient:
    ///   ∂F_l[i]/∂Π_l[i] = ½ residual_l[i]² − ½/Π_l[i]
    ///
    /// Update:
    ///   Π_l[i] ← max(ε, Π_l[i] − η · ∂F_l[i]/∂Π_l[i])
    ///
    /// MLE fixed point: Π_l[i]* = 1 / residual_l[i]²
    /// </summary>
    public void UpdateDiagonalPrecision(float eta)
    {
        if (DiagonalPrecision == null)
            throw new InvalidOperationException(
                "Call InitDiagonalPrecision() before UpdateDiagonalPrecision().");

        for (int i = 0; i < Size; i++)
        {
            float pi = DiagonalPrecision[i];
            float residualSq = (pi > 0f) ? (PredictionError[i] / pi) * (PredictionError[i] / pi) : 0f;
            float grad = 0.5f * residualSq - 0.5f / pi;
            DiagonalPrecision[i] = MathF.Max(1e-6f, pi - eta * grad);
        }
    }

    // ── Clone ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Deep-copies this layer: all tensor buffers are cloned so the
    /// returned instance is fully independent from the original.
    /// Used by <see cref="LayerStore"/> to serve <see cref="LayerStore.ReadLayer"/>
    /// without serialization/deserialization when lazy persistence is enabled.
    /// </summary>
    internal Layer Clone()
    {
        var c = new Layer
        {
            Index = Index,
            Size = Size,
            LowerSize = LowerSize,
            StateLearningRate = StateLearningRate,
            WeightLearningRate = WeightLearningRate,
            WeightDecay = WeightDecay,
            ActivationType = ActivationType,
            Precision = Precision,
            IsClamped = IsClamped,
            Mu = Mu.Clone(),
            PredictionError = PredictionError.Clone(),
        };
        if (GenerativeWeights != null) c.GenerativeWeights = GenerativeWeights.Clone();
        if (Bias != null) c.Bias = Bias.Clone();
        if (PreActivation != null) c.PreActivation = PreActivation.Clone();
        if (Prediction != null) c.Prediction = Prediction.Clone();
        if (DiagonalPrecision != null) c.DiagonalPrecision = DiagonalPrecision.Clone();
        if (PriorMean != null) c.PriorMean = PriorMean.Clone();
        return c;
    }

    // ── Utilities ─────────────────────────────────────────────────────────────

    public float WeightNorm() =>
        GenerativeWeights == null ? 0f : MathF.Sqrt(TensorUtils.SumSquares(GenerativeWeights));

    /// <summary>
    /// Local free-energy contribution.
    /// Scalar precision:   F_l = ½ ||ε_l||² / Π_l
    /// Diagonal precision: F_l = ½ Σ_i ε_l[i]² / Π_l[i]
    /// (Bogacz 2017, eq. 1; Huang &amp; Rao 2011, diagonal case)
    /// </summary>
    public float LocalEnergy()
    {
        if (DiagonalPrecision != null)
        {
            float sum = 0f;
            for (int i = 0; i < Size; i++)
            {
                float pi = DiagonalPrecision[i];
                sum += (pi > 0f) ? PredictionError[i] * PredictionError[i] / pi : 0f;
            }
            return 0.5f * sum;
        }
        return 0.5f * TensorUtils.SumSquares(PredictionError) / Precision;
    }

    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        sb.Append($"PCLayer[{Index}] Size={Size}");
        sb.Append(LowerSize > 0 ? $" LowerSize={LowerSize}" : " (input)");
        sb.Append($" Act={ActivationType}");
        if (DiagonalPrecision != null) sb.Append(" [diag-Π]");
        else if (Precision != 1f) sb.Append($" Π={Precision:F3}");
        if (WeightDecay > 0f) sb.Append($" λ={WeightDecay:G3}");
        if (PriorMean != null) sb.Append(" [has prior]");
        if (IsClamped) sb.Append(" [CLAMPED]");
        return sb.ToString();
    }
}