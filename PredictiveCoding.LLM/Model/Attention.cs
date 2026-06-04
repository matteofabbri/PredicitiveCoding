using PredictiveCoding.Core;

namespace PredictiveCoding.LLM.Model;

// ─────────────────────────────────────────────────────────────────────────────
// Attention  —  causal multi-head self-attention with a key/value cache.
//
// The model decodes one position at a time.  Each step appends the new key and
// value vectors (per head) to the cache and attends the current query over all
// cached positions.  Because the cache only ever contains the current and past
// positions, causality is enforced automatically — no explicit mask is needed.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Per-layer key/value cache: one growing list of K and V vectors per head.</summary>
public sealed class KvCacheLayer
{
    private readonly List<float[]>[] _k;
    private readonly List<float[]>[] _v;

    public KvCacheLayer(int numHeads)
    {
        _k = new List<float[]>[numHeads];
        _v = new List<float[]>[numHeads];
        for (int h = 0; h < numHeads; h++) { _k[h] = []; _v[h] = []; }
    }

    public int Length => _k[0].Count;

    public void Append(int head, float[] k, float[] v) { _k[head].Add(k); _v[head].Add(v); }
    public float[] Key(int head, int pos)   => _k[head][pos];
    public float[] Value(int head, int pos) => _v[head][pos];
}

/// <summary>Key/value cache across all transformer layers for one generation.</summary>
public sealed class KvCache
{
    public KvCacheLayer[] Layers { get; }

    public KvCache(int numLayers, int numHeads)
    {
        Layers = new KvCacheLayer[numLayers];
        for (int l = 0; l < numLayers; l++) Layers[l] = new KvCacheLayer(numHeads);
    }

    /// <summary>Number of positions currently cached.</summary>
    public int Length => Layers.Length == 0 ? 0 : Layers[0].Length;
}

public sealed class MultiHeadAttention
{
    private readonly Linear _q, _k, _v, _o;
    private readonly int    _numHeads;
    private readonly int    _headDim;
    private readonly float  _scale;

    public MultiHeadAttention(TransformerConfig cfg, Random rng)
    {
        _numHeads = cfg.NumHeads;
        _headDim  = cfg.HeadDim;
        _scale    = 1f / MathF.Sqrt(_headDim);

        _q = new Linear(cfg.EmbedDim, cfg.EmbedDim, rng);
        _k = new Linear(cfg.EmbedDim, cfg.EmbedDim, rng);
        _v = new Linear(cfg.EmbedDim, cfg.EmbedDim, rng);
        _o = new Linear(cfg.EmbedDim, cfg.EmbedDim, rng);
    }

    /// <summary>
    /// Attends the current token <paramref name="x"/> over the cached context,
    /// appending its own key/value first.  Returns the attention output (EmbedDim).
    /// </summary>
    public DenseTensor<float> Forward(DenseTensor<float> x, KvCacheLayer cache)
    {
        float[] q = _q.Forward(x).Buffer;
        float[] k = _k.Forward(x).Buffer;
        float[] v = _v.Forward(x).Buffer;

        int embed = _numHeads * _headDim;
        var ctx   = new DenseTensor<float>(embed);
        var ctxB  = ctx.Buffer;

        for (int h = 0; h < _numHeads; h++)
        {
            int off = h * _headDim;

            // Append this position's key/value for the head.
            var kh = new float[_headDim];
            var vh = new float[_headDim];
            Array.Copy(k, off, kh, 0, _headDim);
            Array.Copy(v, off, vh, 0, _headDim);
            cache.Append(h, kh, vh);

            int n      = cache.Length;
            var scores = new float[n];
            for (int p = 0; p < n; p++)
                scores[p] = LlmMath.Dot(q, off, cache.Key(h, p), 0, _headDim) * _scale;
            LlmMath.SoftmaxInPlace(scores, n);

            // Weighted sum of values → context for this head.
            for (int p = 0; p < n; p++)
            {
                float w  = scores[p];
                var   vp = cache.Value(h, p);
                for (int i = 0; i < _headDim; i++) ctxB[off + i] += w * vp[i];
            }
        }

        return _o.Forward(ctx);
    }
}
