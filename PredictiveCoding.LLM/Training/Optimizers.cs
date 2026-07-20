using PredictiveCoding.LLM.Autograd;

namespace PredictiveCoding.LLM.Training;

// ─────────────────────────────────────────────────────────────────────────────
// Optimizers  —  Adam / AdamW parameter updates and global-norm gradient clipping.
//
// Operates on the autograd Tensor parameters (Data + Grad). AdamW uses decoupled
// weight decay (Loshchilov & Hutter 2019); set WeightDecay = 0 for plain Adam.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class Adam
{
    private readonly float _beta1, _beta2, _eps, _weightDecay;
    private readonly bool  _decoupled;                       // true ⇒ AdamW
    private readonly Dictionary<Tensor, (float[] m, float[] v)> _state = new();
    private int _t;

    public Adam(float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
                float weightDecay = 0f, bool decoupled = true)
    {
        _beta1 = beta1; _beta2 = beta2; _eps = eps;
        _weightDecay = weightDecay; _decoupled = decoupled;
    }

    /// <summary>Applies one update step to <paramref name="parameters"/> at learning rate <paramref name="lr"/>.</summary>
    public void Step(IReadOnlyList<Tensor> parameters, float lr)
    {
        _t++;
        float bc1 = 1f - MathF.Pow(_beta1, _t);
        float bc2 = 1f - MathF.Pow(_beta2, _t);

        foreach (var p in parameters)
        {
            if (!_state.TryGetValue(p, out var s))
            {
                s = (new float[p.Data.Length], new float[p.Data.Length]);
                _state[p] = s;
            }
            var (m, v) = s;

            for (int i = 0; i < p.Data.Length; i++)
            {
                float g = p.Grad[i];
                if (_weightDecay != 0f && !_decoupled) g += _weightDecay * p.Data[i];

                m[i] = _beta1 * m[i] + (1f - _beta1) * g;
                v[i] = _beta2 * v[i] + (1f - _beta2) * g * g;

                float mHat = m[i] / bc1;
                float vHat = v[i] / bc2;
                p.Data[i] -= lr * mHat / (MathF.Sqrt(vHat) + _eps);

                if (_weightDecay != 0f && _decoupled) p.Data[i] -= lr * _weightDecay * p.Data[i];
            }
        }
    }

    public static void ZeroGrad(IReadOnlyList<Tensor> parameters)
    {
        foreach (var p in parameters) Array.Clear(p.Grad);
    }

    // ── State access for checkpoint/resume (roadmap-Pro Phase 5) ──────────────

    /// <summary>The bias-correction step counter <c>t</c> (0 before the first step).</summary>
    public int StepCount => _t;

    /// <summary>Restores the step counter so a resumed run continues bias correction, not restarts it.</summary>
    public void RestoreStepCount(int t)
    {
        if (t < 0) throw new ArgumentOutOfRangeException(nameof(t));
        _t = t;
    }

    /// <summary>The (m, v) moment vectors for a parameter, or null if it has not been stepped yet.</summary>
    public (float[] m, float[] v)? MomentsFor(Tensor p) => _state.TryGetValue(p, out var s) ? s : null;

    /// <summary>Installs the (m, v) moment vectors for a parameter (lengths must match its data).</summary>
    public void SetMoments(Tensor p, float[] m, float[] v)
    {
        if (m.Length != p.Data.Length || v.Length != p.Data.Length)
            throw new ArgumentException("Moment vectors must match the parameter length.");
        _state[p] = (m, v);
    }
}

public static class GradientClipping
{
    /// <summary>
    /// Scales all gradients so their global L2 norm ≤ <paramref name="maxNorm"/>.
    /// Returns the pre-clip norm.
    /// </summary>
    public static float ClipGlobalNorm(IReadOnlyList<Tensor> parameters, float maxNorm)
    {
        double sq = 0d;
        foreach (var p in parameters)
            for (int i = 0; i < p.Grad.Length; i++)
            {
                // NaN/±Inf → 0: un gradiente non-finito non deve mai entrare nei pesi.
                // Il clip di norma da solo non basta (con un NaN la norma e' NaN e
                // `norm > maxNorm` e' false, quindi salterebbe).
                if (!float.IsFinite(p.Grad[i])) p.Grad[i] = 0f;
                sq += (double)p.Grad[i] * p.Grad[i];
            }
        float norm = (float)Math.Sqrt(sq);

        if (norm > maxNorm && norm > 0f)
        {
            float scale = maxNorm / (norm + 1e-6f);
            foreach (var p in parameters)
                for (int i = 0; i < p.Grad.Length; i++) p.Grad[i] *= scale;
        }
        return norm;
    }
}
