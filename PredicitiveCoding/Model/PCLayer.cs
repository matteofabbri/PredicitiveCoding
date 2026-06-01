using PredictiveCoding.Core;

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

public sealed class PCLayer
{
    // ── Structure ────────────────────────────────────────────────────────────
    public int Index     { get; init; }   // index in network (0 = sensory)
    public int Size      { get; init; }   // dimensionality of this layer
    public int LowerSize { get; init; }   // dimensionality of the layer below (0 if input)

    // ── Hyperparameters ───────────────────────────────────────────────────────
    public float          StateLearningRate  { get; set; }
    public float          WeightLearningRate { get; set; }
    public ActivationType ActivationType    { get; init; }

    // ── Bogacz (2017) precision and prior (eq. 6) ─────────────────────────────

    /// <summary>
    /// Π_l : isotropic precision (inverse variance) for this layer's prediction error.
    /// Scales ε_l = Π_l · (μ_l − prediction).  Default = 1 (unit precision).
    ///
    /// Setting Precision on the sensory layer (Π_x) vs. the top layer (Π_v)
    /// controls the relative trust in data vs. prior (Bogacz eq. 11):
    ///   v* = (θ^T Π_x θ + Π_v)^{-1} θ^T Π_x x
    /// </summary>
    public float Precision { get; set; } = 1f;

    /// <summary>
    /// v_p : prior mean for the top layer (Bogacz eq. 6: ε_v = Π_v (v − v_p)).
    /// Null (default) → flat prior (ε_top = 0, MLE inference).
    /// Set on the top-most layer to obtain proper Bayesian inference.
    /// </summary>
    public DenseTensor<float>? PriorMean { get; set; }

    // ── Flag ─────────────────────────────────────────────────────────────────
    public bool IsClamped { get; set; }

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

    public void Initialize(Random rng)
    {
        Mu              = RandomUtils.SampleNormal([Size], 0f, 0.01f, rng);
        PredictionError = DenseTensor<float>.Zeros(Size);

        if (LowerSize == 0) return;

        GenerativeWeights = RandomUtils.InitWeights(Size, LowerSize, ActivationType, rng);
        Bias              = RandomUtils.ZeroBias(LowerSize);
        PreActivation     = DenseTensor<float>.Zeros(LowerSize);
        Prediction        = DenseTensor<float>.Zeros(LowerSize);
    }

    // ── Forward: top-down prediction ─────────────────────────────────────────

    /// <summary>pred_l = f(W_l μ_l + b_l)   (Rao &amp; Ballard eq. 2)</summary>
    public void ComputePrediction()
    {
        if (GenerativeWeights == null) return;

        var pre = TensorUtils.MatVecMul(GenerativeWeights, Mu);
        TensorUtils.InPlaceAdd(Bias!, pre);
        pre.CopyTo(PreActivation!);
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
        if (Precision == 1f)
            residual.CopyTo(PredictionError);
        else
            TensorUtils.ScalarMul(residual, Precision).CopyTo(PredictionError);
    }

    // ── State update (inference step) ────────────────────────────────────────

    /// <summary>
    /// μ_l ← μ_l − η_s (ε_l − W_l^T [f'(pre_l) ⊙ ε_{l-1}])
    ///
    /// Because ε already carries the precision weighting (eq. 6), this gradient
    /// automatically implements the full Bogacz eq. 5:
    ///   dv/dt = g'(v)^T ε_x − ε_v
    /// (Rao &amp; Ballard eq. 2.3; Whittington &amp; Bogacz eq. 14)
    /// </summary>
    public void UpdateState(DenseTensor<float>? lowerLayerError, float etaState)
    {
        if (IsClamped || GenerativeWeights == null) return;

        DenseTensor<float> grad;
        if (lowerLayerError != null)
        {
            var fPrime  = ActivationFn.Derivative(PreActivation!, Prediction, ActivationType);
            var delta   = TensorUtils.ElementwiseMul(fPrime, lowerLayerError);
            var wTDelta = TensorUtils.MatVecMulT(GenerativeWeights, delta);
            grad = TensorUtils.ElementwiseSub(PredictionError, wTDelta);
        }
        else
        {
            grad = PredictionError.Clone();
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

        var fPrime   = ActivationFn.Derivative(PreActivation!, Prediction, ActivationType);
        var delta    = TensorUtils.ElementwiseMul(fPrime, lowerLayerError);
        var deltaW   = TensorUtils.OuterProduct(delta, Mu);
        TensorUtils.InPlaceAxpy(etaWeight, deltaW, GenerativeWeights);
        TensorUtils.InPlaceAxpy(etaWeight, delta,  Bias!);
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

    // ── Utilities ─────────────────────────────────────────────────────────────

    public float WeightNorm() =>
        GenerativeWeights == null ? 0f : MathF.Sqrt(TensorUtils.SumSquares(GenerativeWeights));

    /// <summary>
    /// Local free-energy contribution F_l = ½ Π_l ||residual||² = ½ ||ε_l||² / Π_l
    /// (Bogacz 2017, eq. 1 extended to multiple layers)
    /// </summary>
    public float LocalEnergy() =>
        0.5f * TensorUtils.SumSquares(PredictionError) / Precision;

    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        sb.Append($"PCLayer[{Index}] Size={Size}");
        sb.Append(LowerSize > 0 ? $" LowerSize={LowerSize}" : " (input)");
        sb.Append($" Act={ActivationType}");
        if (Precision != 1f)  sb.Append($" Π={Precision:F3}");
        if (PriorMean != null) sb.Append(" [has prior]");
        if (IsClamped)         sb.Append(" [CLAMPED]");
        return sb.ToString();
    }
}
