using PredictiveCoding.Core;

namespace PredictiveCoding.Model;

// ─────────────────────────────────────────────────────────────────────────────
// PCLayer  —  Single layer in a hierarchical predictive coding network
//
// Ogni layer l mantiene:
//   Id         : Guid V7 (time-ordered, univoco per sessione/runtime)
//   μ_l        : stato latente corrente  [Size]
//   W_l        : pesi generativi         [LowerSize × Size]   (null se layer input)
//   b_l        : bias generativo         [LowerSize]           (null se layer input)
//   pre_l      : pre-attivazione         W_l μ_l + b_l  [LowerSize]
//   pred_l     : predizione top-down     f(pre_l)        [LowerSize]
//   ε_l        : errore di predizione    μ_l − pred_{l+1}→l  [Size]
//
// Riferimenti:
//   Rao & Ballard (1999); Whittington & Bogacz (2017)
// ─────────────────────────────────────────────────────────────────────────────

public sealed class PCLayer
{
    // ── Struttura ────────────────────────────────────────────────────────────
    public int Index     { get; init; }   // indice nella rete (0 = sensory)
    public int Size      { get; init; }   // dimensione di questo layer
    public int LowerSize { get; init; }   // dimensione del layer sotto (0 se input)

    // ── Iperparametri ─────────────────────────────────────────────────────────
    public float          StateLearningRate  { get; set; }
    public float          WeightLearningRate { get; set; }
    public ActivationType ActivationType    { get; init; }

    // ── Flag ─────────────────────────────────────────────────────────────────
    public bool IsClamped { get; set; }

    // ── Stato latente ─────────────────────────────────────────────────────────
    /// <summary>μ_l : stato latente corrente  [Size]</summary>
    public DenseTensor<float> Mu { get; internal set; } = null!;

    // ── Modello generativo (null per il layer di input) ───────────────────────
    /// <summary>W_l : pesi generativi  [LowerSize × Size]</summary>
    public DenseTensor<float>? GenerativeWeights { get; internal set; }

    /// <summary>b_l : bias generativo  [LowerSize]</summary>
    public DenseTensor<float>? Bias { get; internal set; }

    // ── Quantità calcolate ────────────────────────────────────────────────────
    /// <summary>pre_l = W_l μ_l + b_l  [LowerSize]</summary>
    public DenseTensor<float>? PreActivation { get; internal set; }

    /// <summary>pred_l = f(pre_l)  [LowerSize]</summary>
    public DenseTensor<float>? Prediction { get; internal set; }

    /// <summary>ε_l = μ_l - predizione ricevuta da layer l+1  [Size]</summary>
    public DenseTensor<float> PredictionError { get; internal set; } = null!;

    // ── Inizializzazione ─────────────────────────────────────────────────────

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


    // ── Forward: predizione top-down ──────────────────────────────────────────

    /// <summary>
    /// pred_l = f(W_l μ_l + b_l)   (Rao & Ballard eq. 2)
    /// </summary>
    public void ComputePrediction()
    {
        if (GenerativeWeights == null) return;

        var pre = TensorUtils.MatVecMul(GenerativeWeights, Mu);
        TensorUtils.InPlaceAdd(Bias!, pre);
        pre.CopyTo(PreActivation!);
        ActivationFn.Apply(pre, ActivationType).CopyTo(Prediction!);
    }

    // ── Errore di predizione ──────────────────────────────────────────────────

    /// <summary>ε_l = μ_l − pred_from_above</summary>
    public void SetPredictionError(DenseTensor<float>? predFromAbove)
    {
        if (predFromAbove == null)
        {
            PredictionError.Zero();
            return;
        }
        ShapeCheck.SameLength(Mu, predFromAbove);
        TensorUtils.ElementwiseSub(Mu, predFromAbove).CopyTo(PredictionError);
    }

    // ── Aggiornamento stato (inference step) ──────────────────────────────────

    /// <summary>
    /// μ_l ← μ_l − η_s (ε_l − W_l^T [f'(pre_l) ⊙ ε_{l-1}])
    /// (Rao & Ballard eq. 2.3; Whittington & Bogacz eq. 14)
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

    // ── Aggiornamento pesi (learning step) ───────────────────────────────────

    /// <summary>
    /// W_l ← W_l + η_W (f'(pre_l) ⊙ ε_{l-1}) ⊗ μ_l^T
    /// (Whittington & Bogacz eq. 20; regola di Hebb pesata dall'errore)
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

    // ── Utilità ───────────────────────────────────────────────────────────────

    public float WeightNorm() =>
        GenerativeWeights == null ? 0f : MathF.Sqrt(TensorUtils.SumSquares(GenerativeWeights));

    public float LocalEnergy() => 0.5f * TensorUtils.SumSquares(PredictionError);

    public override string ToString() =>
        $"PCLayer[{Index}] Size={Size}" +
        (LowerSize > 0 ? $" LowerSize={LowerSize}" : " (input)") +
        $" Act={ActivationType}" +
        (IsClamped ? " [CLAMPED]" : "");
}
