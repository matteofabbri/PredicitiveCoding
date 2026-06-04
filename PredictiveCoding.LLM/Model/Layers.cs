using PredictiveCoding.Core;

namespace PredictiveCoding.LLM.Model;

// ─────────────────────────────────────────────────────────────────────────────
// Layers  —  the elementary building blocks of the transformer.
//
// All activations are represented as a single rank-1 DenseTensor<float> of length
// EmbedDim (the model processes one token position at a time, which pairs
// naturally with the KV cache used during generation).
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Small numeric helpers operating on raw float buffers.</summary>
internal static class LlmMath
{
    public static float Dot(float[] a, int ao, float[] b, int bo, int n)
    {
        float s = 0f;
        for (int i = 0; i < n; i++) s += a[ao + i] * b[bo + i];
        return s;
    }

    public static int Argmax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    /// <summary>Numerically-stable softmax over the first <paramref name="n"/> entries, in place.</summary>
    public static void SoftmaxInPlace(float[] v, int n)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++) if (v[i] > max) max = v[i];
        float sum = 0f;
        for (int i = 0; i < n; i++) { v[i] = MathF.Exp(v[i] - max); sum += v[i]; }
        float inv = sum > 0f ? 1f / sum : 0f;
        for (int i = 0; i < n; i++) v[i] *= inv;
    }

    /// <summary>GELU activation (tanh approximation), in place.</summary>
    public static void GeluInPlace(float[] v)
    {
        const float c = 0.7978845608f; // sqrt(2/π)
        for (int i = 0; i < v.Length; i++)
        {
            float x = v[i];
            v[i] = 0.5f * x * (1f + MathF.Tanh(c * (x + 0.044715f * x * x * x)));
        }
    }

    /// <summary>SiLU / Swish activation  x·σ(x), in place (used by SwiGLU).</summary>
    public static void SiluInPlace(float[] v)
    {
        for (int i = 0; i < v.Length; i++)
        {
            float x = v[i];
            v[i] = x / (1f + MathF.Exp(-x));
        }
    }

    /// <summary>RMS-normalises one head slice [offset, offset+dim) in place (QK-Norm core).</summary>
    public static void RmsNormSlice(float[] buf, int offset, int dim, float eps)
    {
        float ms = 0f;
        for (int i = 0; i < dim; i++) { float x = buf[offset + i]; ms += x * x; }
        float inv = 1f / MathF.Sqrt(ms / dim + eps);
        for (int i = 0; i < dim; i++) buf[offset + i] *= inv;
    }
}

/// <summary>Dense linear projection  y = W·x + b,  W ∈ ℝ^{out×in}.</summary>
public sealed class Linear
{
    private readonly DenseTensor<float> _w;   // [out, in]
    private readonly DenseTensor<float> _b;   // [out]

    public int InputDim  => _w.Dimensions[1];
    public int OutputDim => _w.Dimensions[0];

    public Linear(int inDim, int outDim, Random rng, ActivationType init = ActivationType.Linear)
    {
        _w = RandomUtils.InitWeights(inDim, outDim, init, rng);   // → [out, in]
        _b = DenseTensor<float>.Zeros(outDim);
    }

    public DenseTensor<float> Forward(DenseTensor<float> x)
        => TensorUtils.ElementwiseAdd(TensorUtils.MatVecMul(_w, x), _b);
}

/// <summary>Normalisation layer abstraction (LayerNorm or RMSNorm).</summary>
public interface INormLayer
{
    DenseTensor<float> Forward(DenseTensor<float> x);
}

/// <summary>Creates the normalisation layer selected by the config.</summary>
public static class Norms
{
    public static INormLayer Create(TransformerConfig cfg) => cfg.Norm switch
    {
        NormKind.RmsNorm => new RmsNorm(cfg.EmbedDim, cfg.LayerNormEps),
        _                => new LayerNorm(cfg.EmbedDim, cfg.LayerNormEps),
    };
}

/// <summary>Root-mean-square normalisation (Zhang &amp; Sennrich 2019): scale only, no mean subtraction.</summary>
public sealed class RmsNorm : INormLayer
{
    private readonly float[] _gamma;
    private readonly float   _eps;
    private readonly int     _dim;

    public RmsNorm(int dim, float eps)
    {
        _dim   = dim;
        _eps   = eps;
        _gamma = new float[dim];
        Array.Fill(_gamma, 1f);
    }

    public DenseTensor<float> Forward(DenseTensor<float> x)
    {
        var src = x.Buffer;
        float ms = 0f;
        for (int i = 0; i < _dim; i++) ms += src[i] * src[i];
        ms /= _dim;
        float invRms = 1f / MathF.Sqrt(ms + _eps);

        var outT = new DenseTensor<float>(_dim);
        var dst  = outT.Buffer;
        for (int i = 0; i < _dim; i++) dst[i] = src[i] * invRms * _gamma[i];
        return outT;
    }
}

/// <summary>Rotary position embedding (Su et al. 2021): rotates Q/K head pairs by position-dependent angles.</summary>
public static class RotaryEmbedding
{
    /// <summary>Rotates one head slice (length headDim, even) in place for the given position.</summary>
    public static void Rotate(float[] buf, int offset, int headDim, int position)
    {
        int half = headDim / 2;
        for (int i = 0; i < half; i++)
        {
            float freq  = 1f / MathF.Pow(10000f, (float)(2 * i) / headDim);
            float angle = position * freq;
            float cos   = MathF.Cos(angle), sin = MathF.Sin(angle);
            int   a = offset + 2 * i, b = a + 1;
            float x0 = buf[a], x1 = buf[b];
            buf[a] = x0 * cos - x1 * sin;
            buf[b] = x0 * sin + x1 * cos;
        }
    }
}

/// <summary>Layer normalisation with learnable gain (γ) and bias (β).</summary>
public sealed class LayerNorm : INormLayer
{
    private readonly float[] _gamma;
    private readonly float[] _beta;
    private readonly float   _eps;
    private readonly int     _dim;

    public LayerNorm(int dim, float eps)
    {
        _dim   = dim;
        _eps   = eps;
        _gamma = new float[dim];
        _beta  = new float[dim];
        Array.Fill(_gamma, 1f);
    }

    public DenseTensor<float> Forward(DenseTensor<float> x)
    {
        var src = x.Buffer;
        float mean = 0f;
        for (int i = 0; i < _dim; i++) mean += src[i];
        mean /= _dim;

        float var = 0f;
        for (int i = 0; i < _dim; i++) { float d = src[i] - mean; var += d * d; }
        var /= _dim;
        float invStd = 1f / MathF.Sqrt(var + _eps);

        var outT = new DenseTensor<float>(_dim);
        var dst  = outT.Buffer;
        for (int i = 0; i < _dim; i++)
            dst[i] = (src[i] - mean) * invStd * _gamma[i] + _beta[i];
        return outT;
    }
}

/// <summary>Token embedding table with an optional tied output projection.</summary>
public sealed class Embedding
{
    private readonly DenseTensor<float> _table;   // [vocab, dim]
    private readonly int _vocab;
    private readonly int _dim;

    public Embedding(int vocab, int dim, Random rng)
    {
        _vocab = vocab;
        _dim   = dim;
        _table = RandomUtils.SampleNormal([vocab, dim], 0f, 0.02f, rng);
    }

    /// <summary>Returns a fresh copy of the embedding row for <paramref name="token"/>.</summary>
    public DenseTensor<float> Lookup(int token)
    {
        if (token < 0 || token >= _vocab)
            throw new ArgumentOutOfRangeException(nameof(token), $"token {token} out of range [0,{_vocab}).");
        var v = new DenseTensor<float>(_dim);
        Array.Copy(_table.Buffer, token * _dim, v.Buffer, 0, _dim);
        return v;
    }

    /// <summary>Tied LM head: logits[v] = ⟨emb_v, x⟩ for every vocabulary item v.</summary>
    public DenseTensor<float> Project(DenseTensor<float> x)
    {
        var logits = new DenseTensor<float>(_vocab);
        for (int v = 0; v < _vocab; v++)
            logits.Buffer[v] = LlmMath.Dot(_table.Buffer, v * _dim, x.Buffer, 0, _dim);
        return logits;
    }
}

/// <summary>Fixed sinusoidal positional encoding (Vaswani et al. 2017).</summary>
public sealed class PositionalEncoding
{
    private readonly int _dim;

    public PositionalEncoding(int dim) => _dim = dim;

    /// <summary>Adds the encoding for <paramref name="position"/> to <paramref name="x"/> in place.</summary>
    public void AddTo(DenseTensor<float> x, int position)
    {
        var buf = x.Buffer;
        for (int i = 0; i < _dim; i += 2)
        {
            float freq  = 1f / MathF.Pow(10000f, (float)i / _dim);
            float angle = position * freq;
            buf[i] += MathF.Sin(angle);
            if (i + 1 < _dim) buf[i + 1] += MathF.Cos(angle);
        }
    }
}
