using PredictiveCoding.Core;

namespace PredictiveCoding.LLM.Model;

// ─────────────────────────────────────────────────────────────────────────────
// Attention  —  causal multi-head self-attention with a key/value cache.
//
// Supports:
//   • grouped-query / multi-query attention — NumKvHeads ≤ NumHeads query heads
//     share key/value heads (NumKvHeads = 1 ⇒ MQA, = NumHeads ⇒ standard MHA);
//   • rotary position embedding (RoPE) applied to Q/K;
//   • flash-attention-style online softmax (single pass, no full score vector).
//
// The model decodes one position at a time, so causality is implicit: the cache
// only ever holds the current and past positions.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Per-layer key/value cache: one growing list of K and V vectors per KV head.</summary>
public sealed class KvCacheLayer
{
    private readonly List<float[]>[] _k;
    private readonly List<float[]>[] _v;

    public KvCacheLayer(int numKvHeads)
    {
        _k = new List<float[]>[numKvHeads];
        _v = new List<float[]>[numKvHeads];
        for (int h = 0; h < numKvHeads; h++) { _k[h] = []; _v[h] = []; }
    }

    public int Length => _k[0].Count;

    public void Append(int head, float[] k, float[] v) { _k[head].Add(k); _v[head].Add(v); }
    public float[] Key(int head, int pos)   => _k[head][pos];
    public float[] Value(int head, int pos) => _v[head][pos];

    /// <summary>Deep copy (used by beam search to fork a hypothesis).</summary>
    public KvCacheLayer Clone()
    {
        var c = new KvCacheLayer(_k.Length);
        for (int h = 0; h < _k.Length; h++)
        {
            foreach (var k in _k[h]) c._k[h].Add((float[])k.Clone());
            foreach (var v in _v[h]) c._v[h].Add((float[])v.Clone());
        }
        return c;
    }
}

/// <summary>Key/value cache across all transformer layers for one generation.</summary>
public sealed class KvCache
{
    public KvCacheLayer[] Layers { get; }

    public KvCache(int numLayers, int numKvHeads)
    {
        Layers = new KvCacheLayer[numLayers];
        for (int l = 0; l < numLayers; l++) Layers[l] = new KvCacheLayer(numKvHeads);
    }

    private KvCache(KvCacheLayer[] layers) => Layers = layers;

    /// <summary>Number of positions currently cached.</summary>
    public int Length => Layers.Length == 0 ? 0 : Layers[0].Length;

    /// <summary>Deep copy (used by beam search to fork a hypothesis).</summary>
    public KvCache Clone()
    {
        var layers = new KvCacheLayer[Layers.Length];
        for (int l = 0; l < Layers.Length; l++) layers[l] = Layers[l].Clone();
        return new KvCache(layers);
    }
}

public sealed class MultiHeadAttention
{
    private readonly Linear _q, _k, _v, _o;
    private readonly int    _numHeads;
    private readonly int    _numKvHeads;
    private readonly int    _headDim;
    private readonly int    _kvGroup;
    private readonly float   _scale;
    private readonly bool    _rotary;
    private readonly bool    _flash;
    private readonly bool    _alibi;
    private readonly float[] _slopes;

    public MultiHeadAttention(TransformerConfig cfg, Random rng)
    {
        _numHeads   = cfg.NumHeads;
        _numKvHeads = cfg.KvHeads;
        _headDim    = cfg.HeadDim;
        _kvGroup    = cfg.KvGroupSize;
        _scale      = 1f / MathF.Sqrt(_headDim);
        _rotary     = cfg.Positional == PositionalKind.Rotary;
        _flash      = cfg.UseFlashAttention;
        _alibi      = cfg.Positional == PositionalKind.Alibi;

        _slopes = new float[_numHeads];                       // geometric ALiBi slopes
        for (int h = 0; h < _numHeads; h++)
            _slopes[h] = MathF.Pow(2f, -(8f / _numHeads) * (h + 1));

        int qDim  = _numHeads   * _headDim;   // = EmbedDim
        int kvDim = _numKvHeads * _headDim;
        _q = new Linear(cfg.EmbedDim, qDim,  rng);
        _k = new Linear(cfg.EmbedDim, kvDim, rng);
        _v = new Linear(cfg.EmbedDim, kvDim, rng);
        _o = new Linear(qDim, cfg.EmbedDim,  rng);
    }

    /// <summary>
    /// Attends the current token <paramref name="x"/> at <paramref name="position"/>
    /// over the cached context, appending its own key/value first.
    /// </summary>
    public DenseTensor<float> Forward(DenseTensor<float> x, KvCacheLayer cache, int position)
    {
        float[] q = _q.Forward(x).Buffer;   // [numHeads·headDim]
        float[] k = _k.Forward(x).Buffer;   // [numKvHeads·headDim]
        float[] v = _v.Forward(x).Buffer;

        if (_rotary)
        {
            for (int h = 0; h < _numHeads;   h++) RotaryEmbedding.Rotate(q, h * _headDim, _headDim, position);
            for (int g = 0; g < _numKvHeads; g++) RotaryEmbedding.Rotate(k, g * _headDim, _headDim, position);
        }

        // Append this position's key/value, once per KV head.
        for (int g = 0; g < _numKvHeads; g++)
        {
            var kh = new float[_headDim];
            var vh = new float[_headDim];
            Array.Copy(k, g * _headDim, kh, 0, _headDim);
            Array.Copy(v, g * _headDim, vh, 0, _headDim);
            cache.Append(g, kh, vh);
        }

        var ctx  = new DenseTensor<float>(_numHeads * _headDim);
        var ctxB = ctx.Buffer;
        int n    = cache.Length;

        for (int h = 0; h < _numHeads; h++)
        {
            int off = h * _headDim;
            int g   = h / _kvGroup;               // which KV head this query head reads
            if (_flash) AttendOnline(q, off, cache, g, n, ctxB, off, h);
            else        AttendStandard(q, off, cache, g, n, ctxB, off, h);
        }

        return _o.Forward(ctx);
    }

    // Standard: materialise scores, softmax, weighted sum.
    private void AttendStandard(float[] q, int qoff, KvCacheLayer cache, int g, int n, float[] outB, int outOff, int head)
    {
        var scores = new float[n];
        for (int p = 0; p < n; p++)
        {
            float s = LlmMath.Dot(q, qoff, cache.Key(g, p), 0, _headDim) * _scale;
            if (_alibi) s -= _slopes[head] * (n - 1 - p);     // linear distance bias
            scores[p] = s;
        }
        LlmMath.SoftmaxInPlace(scores, n);
        for (int p = 0; p < n; p++)
        {
            float w  = scores[p];
            var   vp = cache.Value(g, p);
            for (int i = 0; i < _headDim; i++) outB[outOff + i] += w * vp[i];
        }
    }

    // Flash-style: online softmax in a single pass (running max / sum / accumulator).
    private void AttendOnline(float[] q, int qoff, KvCacheLayer cache, int g, int n, float[] outB, int outOff, int head)
    {
        float m = float.NegativeInfinity, l = 0f;
        Span<float> acc = stackalloc float[_headDim];
        acc.Clear();

        for (int p = 0; p < n; p++)
        {
            float s     = LlmMath.Dot(q, qoff, cache.Key(g, p), 0, _headDim) * _scale;
            if (_alibi) s -= _slopes[head] * (n - 1 - p);
            float mNew  = MathF.Max(m, s);
            float scale = MathF.Exp(m - mNew);   // 0 on the first step (exp(-inf)) → fine
            float pExp  = MathF.Exp(s - mNew);
            l = l * scale + pExp;
            var vp = cache.Value(g, p);
            for (int i = 0; i < _headDim; i++) acc[i] = acc[i] * scale + pExp * vp[i];
            m = mNew;
        }

        float inv = l > 0f ? 1f / l : 0f;
        for (int i = 0; i < _headDim; i++) outB[outOff + i] = acc[i] * inv;
    }
}
