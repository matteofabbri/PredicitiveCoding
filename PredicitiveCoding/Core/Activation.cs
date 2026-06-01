using System.Numerics.Tensors;

namespace PredictiveCoding.Core;

// ─────────────────────────────────────────────────────────────────────────────
// Activation
//
// Funzioni di attivazione non-lineari e loro derivate per il modello generativo.
//
// Nel predictive coding gerarchico (Rao & Ballard 1999, Friston 2005):
//   pred_{l-1} = f(W_l μ_l + b_l)
//
// La derivata f' entra nel gradiente dello stato e dei pesi:
//   δ_l  = f'(pre_l) ⊙ ε_{l-1}          (errore pesato dalla derivata)
//   ∂F/∂μ_l = ε_l - W_l^T δ_l
//   ∂F/∂W_l = -δ_l ⊗ μ_l^T
// ─────────────────────────────────────────────────────────────────────────────

public enum ActivationType
{
    Linear,   // f(x) = x,          f'(x) = 1
    Tanh,     // f(x) = tanh(x),    f'(x) = 1 - tanh(x)^2
    ReLU,     // f(x) = max(0,x),   f'(x) = x>0 ? 1 : 0
    Sigmoid   // f(x) = σ(x),       f'(x) = σ(x)(1-σ(x))
}

public static class ActivationFn
{
    // ── Applicazione ─────────────────────────────────────────────────────────

    /// <summary>
    /// Applica la funzione di attivazione element-wise:  z = f(x)
    /// </summary>
    public static DenseTensor<float> Apply(DenseTensor<float> preActivation, ActivationType type)
    {
        var z = DenseTensor<float>.Like(preActivation);
        var src = preActivation.Buffer.AsSpan();
        var dst = z.Buffer.AsSpan();

        switch (type)
        {
            case ActivationType.Linear:
                // f(x) = x
                src.CopyTo(dst);
                break;

            case ActivationType.Tanh:
                // f(x) = tanh(x)
                TensorPrimitives.Tanh(src, dst);
                break;

            case ActivationType.ReLU:
                // f(x) = max(0, x)
                TensorPrimitives.Max(src, 0f, dst);
                break;

            case ActivationType.Sigmoid:
                // f(x) = 1 / (1 + e^-x)
                TensorPrimitives.Negate(src, dst);
                TensorPrimitives.Exp(dst, dst);
                TensorPrimitives.Add(dst, 1f, dst);
                // dst[i] = 1 / (1 + e^-x[i])
                for (int i = 0; i < dst.Length; i++)
                    dst[i] = 1f / dst[i];
                break;

            default:
                throw new NotSupportedException($"ActivationType non supportato: {type}");
        }

        return z;
    }

    // ── Derivata ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Calcola la derivata element-wise f'(preActivation).
    ///
    /// Per efficienza, quando la derivata può essere espressa tramite il valore
    /// già attivato (es. Tanh, Sigmoid) si usa <paramref name="activated"/>
    /// evitando di ricalcolare f.
    /// </summary>
    /// <param name="preActivation">x = W μ + b (pre-nonlinearità)</param>
    /// <param name="activated">y = f(x) (post-nonlinearità, può essere null per Linear/ReLU)</param>
    /// <param name="type">Tipo di attivazione</param>
    public static DenseTensor<float> Derivative(
        DenseTensor<float> preActivation,
        DenseTensor<float>? activated,
        ActivationType type)
    {
        var d = DenseTensor<float>.Like(preActivation);
        var dst = d.Buffer.AsSpan();

        switch (type)
        {
            case ActivationType.Linear:
                // f'(x) = 1  ∀x
                d.Fill(1f);
                break;

            case ActivationType.Tanh:
                // f'(x) = 1 - tanh(x)^2 = 1 - y^2
                // Usiamo y già calcolato per efficienza
                var y = activated ?? Apply(preActivation, ActivationType.Tanh);
                TensorPrimitives.Multiply(y.Buffer.AsSpan(), y.Buffer.AsSpan(), dst);  // y^2
                TensorPrimitives.Negate(dst, dst);          // -y^2
                TensorPrimitives.Add(dst, 1f, dst);         // 1 - y^2
                break;

            case ActivationType.ReLU:
                // f'(x) = 1 if x > 0, else 0
                var pre = preActivation.Buffer.AsSpan();
                for (int i = 0; i < dst.Length; i++)
                    dst[i] = pre[i] > 0f ? 1f : 0f;
                break;

            case ActivationType.Sigmoid:
                // f'(x) = σ(x)(1 - σ(x)) = y(1 - y)
                var sig = activated ?? Apply(preActivation, ActivationType.Sigmoid);
                var sigSpan = sig.Buffer.AsSpan();
                var tmp = new float[dst.Length];
                TensorPrimitives.Negate(sigSpan, tmp);           // -y
                TensorPrimitives.Add(tmp, 1f, tmp);              // 1 - y
                TensorPrimitives.Multiply(sigSpan, tmp, dst);    // y(1-y)
                break;

            default:
                throw new NotSupportedException($"ActivationType non supportato: {type}");
        }

        return d;
    }

    // ── Softmax (per output di classificazione) ──────────────────────────────

    /// <summary>
    /// Softmax stabile numericamente: σ(x)_i = e^(x_i - max) / Σ_j e^(x_j - max)
    /// Non fa parte della dinamica PC interna; usato solo per visualizzazione.
    /// </summary>
    public static DenseTensor<float> Softmax(DenseTensor<float> x)
    {
        var z = DenseTensor<float>.Like(x);
        float maxVal = TensorPrimitives.Max(x.Buffer.AsSpan());
        TensorPrimitives.Subtract(x.Buffer.AsSpan(), maxVal, z.Buffer.AsSpan());
        TensorPrimitives.Exp(z.Buffer.AsSpan(), z.Buffer.AsSpan());
        float sum = TensorPrimitives.Sum(z.Buffer.AsSpan());
        TensorPrimitives.Divide(z.Buffer.AsSpan(), sum, z.Buffer.AsSpan());
        return z;
    }
}
