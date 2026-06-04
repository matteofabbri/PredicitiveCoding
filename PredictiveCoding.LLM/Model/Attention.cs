using PredictiveCoding.Core;

namespace PredictiveCoding.LLM.Model;

// ─────────────────────────────────────────────────────────────────────────────
// Attention  —  causal self-attention with a per-layer cache.
//
//   • MultiHeadAttention: MHA / GQA / MQA, RoPE or ALiBi, optional QK-Norm
//     (OLMo 2 / Gemma 2 / Qwen3), optional attention-logit soft-capping (Gemma 2),
//     and flash-style online softmax.
//   • LatentAttention: Multi-head Latent Attention (DeepSeek-V2/V3) — the KV cache
//     stores a small compressed latent per token instead of full K/V, with a
//     decoupled-RoPE key.
//
// The model decodes one position at a time, so causality is implicit.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>A per-layer attention cache (key/value or compressed latent).</summary>
public interface ILayerCache
{
    int Length { get; }
    ILayerCache Clone();
}

/// <summary>A self-attention layer plus the cache type it uses.</summary>
public interface ISelfAttention
{
    DenseTensor<float> Forward(DenseTensor<float> x, ILayerCache cache, int position);
    ILayerCache CreateCache();
}

/// <summary>Per-layer key/value cache: one growing list of K and V vectors per KV head.</summary>
public sealed class KvCacheLayer : ILayerCache
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

    public ILayerCache Clone()
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

/// <summary>Per-layer latent cache for MLA: compressed KV latent + decoupled-RoPE key.</summary>
public sealed class LatentKvCacheLayer : ILayerCache
{
    private readonly List<float[]> _ckv = [];   // compressed KV latent c_kv [dc]
    private readonly List<float[]> _kr  = [];   // decoupled-RoPE key k_R [dr]

    public int Length => _ckv.Count;

    public void Append(float[] ckv, float[] kr) { _ckv.Add(ckv); _kr.Add(kr); }
    public float[] Ckv(int pos) => _ckv[pos];
    public float[] Kr(int pos)  => _kr[pos];

    public ILayerCache Clone()
    {
        var c = new LatentKvCacheLayer();
        foreach (var x in _ckv) c._ckv.Add((float[])x.Clone());
        foreach (var x in _kr)  c._kr.Add((float[])x.Clone());
        return c;
    }
}

/// <summary>Key/value cache across all transformer layers for one generation.</summary>
public sealed class KvCache
{
    public ILayerCache[] Layers { get; }

    public KvCache(ILayerCache[] layers) => Layers = layers;

    /// <summary>Number of positions currently cached.</summary>
    public int Length => Layers.Length == 0 ? 0 : Layers[0].Length;

    public KvCache Clone()
    {
        var layers = new ILayerCache[Layers.Length];
        for (int l = 0; l < Layers.Length; l++) layers[l] = Layers[l].Clone();
        return new KvCache(layers);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MultiHeadAttention
// ─────────────────────────────────────────────────────────────────────────────

public sealed class MultiHeadAttention : ISelfAttention
{
    private readonly Linear _q, _k, _v, _o;
    private readonly int    _numHeads, _numKvHeads, _headDim, _kvGroup;
    private readonly float  _scale, _softcap, _eps;
    private readonly bool   _rotary, _flash, _alibi, _qkNorm;
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
        _qkNorm     = cfg.UseQkNorm;
        _softcap    = cfg.AttnLogitSoftcap;
        _eps        = cfg.LayerNormEps;

        _slopes = new float[_numHeads];
        for (int h = 0; h < _numHeads; h++)
            _slopes[h] = MathF.Pow(2f, -(8f / _numHeads) * (h + 1));

        int qDim  = _numHeads   * _headDim;
        int kvDim = _numKvHeads * _headDim;
        _q = new Linear(cfg.EmbedDim, qDim,  rng);
        _k = new Linear(cfg.EmbedDim, kvDim, rng);
        _v = new Linear(cfg.EmbedDim, kvDim, rng);
        _o = new Linear(qDim, cfg.EmbedDim,  rng);
    }

    public ILayerCache CreateCache() => new KvCacheLayer(_numKvHeads);

    public DenseTensor<float> Forward(DenseTensor<float> x, ILayerCache cacheBase, int position)
    {
        var cache = (KvCacheLayer)cacheBase;
        float[] q = _q.Forward(x).Buffer;
        float[] k = _k.Forward(x).Buffer;
        float[] v = _v.Forward(x).Buffer;

        if (_qkNorm)   // RMS-norm each head before RoPE
        {
            for (int h = 0; h < _numHeads;   h++) LlmMath.RmsNormSlice(q, h * _headDim, _headDim, _eps);
            for (int g = 0; g < _numKvHeads; g++) LlmMath.RmsNormSlice(k, g * _headDim, _headDim, _eps);
        }
        if (_rotary)
        {
            for (int h = 0; h < _numHeads;   h++) RotaryEmbedding.Rotate(q, h * _headDim, _headDim, position);
            for (int g = 0; g < _numKvHeads; g++) RotaryEmbedding.Rotate(k, g * _headDim, _headDim, position);
        }

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
            int g   = h / _kvGroup;
            if (_flash) AttendOnline(q, off, cache, g, n, ctxB, off, h);
            else        AttendStandard(q, off, cache, g, n, ctxB, off, h);
        }
        return _o.Forward(ctx);
    }

    private float Score(float[] q, int qoff, float[] key, int n, int p, int head)
    {
        float s = LlmMath.Dot(q, qoff, key, 0, _headDim) * _scale;
        if (_alibi)        s -= _slopes[head] * (n - 1 - p);
        if (_softcap > 0f) s = _softcap * MathF.Tanh(s / _softcap);
        return s;
    }

    private void AttendStandard(float[] q, int qoff, KvCacheLayer cache, int g, int n, float[] outB, int outOff, int head)
    {
        var scores = new float[n];
        for (int p = 0; p < n; p++) scores[p] = Score(q, qoff, cache.Key(g, p), n, p, head);
        LlmMath.SoftmaxInPlace(scores, n);
        for (int p = 0; p < n; p++)
        {
            float w  = scores[p];
            var   vp = cache.Value(g, p);
            for (int i = 0; i < _headDim; i++) outB[outOff + i] += w * vp[i];
        }
    }

    // Flash-style online softmax (single pass, no full score vector).
    private void AttendOnline(float[] q, int qoff, KvCacheLayer cache, int g, int n, float[] outB, int outOff, int head)
    {
        float m = float.NegativeInfinity, l = 0f;
        Span<float> acc = stackalloc float[_headDim];
        acc.Clear();
        for (int p = 0; p < n; p++)
        {
            float s     = Score(q, qoff, cache.Key(g, p), n, p, head);
            float mNew  = MathF.Max(m, s);
            float scale = MathF.Exp(m - mNew);
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

// ─────────────────────────────────────────────────────────────────────────────
// LatentAttention  —  Multi-head Latent Attention (MLA), DeepSeek-V2/V3.
//
// Down-projects the hidden state to a compressed KV latent c_kv (cached), plus a
// decoupled-RoPE key k_R (cached). Per-head content keys/values are reconstructed
// from c_kv on the fly; the attention score combines the content part (q_C·k_C)
// with the decoupled RoPE part (q_R·k_R). The cache stores only c_kv + k_R, so it
// is much smaller than a full KV cache.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class LatentAttention : ISelfAttention
{
    private readonly Linear _wDkv, _wUk, _wUv, _wQ, _wKr, _wQr, _wO;
    private readonly int    _numHeads, _headDim, _dc, _dr;
    private readonly float  _scale, _softcap;

    public LatentAttention(TransformerConfig cfg, Random rng)
    {
        _numHeads = cfg.NumHeads;
        _headDim  = cfg.HeadDim;
        _dc       = cfg.KvLatentDim;
        _dr       = cfg.RopeHeadDim;
        _scale    = 1f / MathF.Sqrt(_headDim + _dr);
        _softcap  = cfg.AttnLogitSoftcap;

        int hd = _numHeads * _headDim;
        _wDkv = new Linear(cfg.EmbedDim, _dc, rng);             // h → c_kv
        _wUk  = new Linear(_dc, hd, rng);                       // c_kv → per-head K content
        _wUv  = new Linear(_dc, hd, rng);                       // c_kv → per-head V
        _wQ   = new Linear(cfg.EmbedDim, hd, rng);              // h → per-head Q content
        _wKr  = new Linear(cfg.EmbedDim, _dr, rng);             // h → decoupled RoPE key (shared)
        _wQr  = new Linear(cfg.EmbedDim, _numHeads * _dr, rng); // h → decoupled RoPE query per head
        _wO   = new Linear(hd, cfg.EmbedDim, rng);
    }

    public ILayerCache CreateCache() => new LatentKvCacheLayer();

    public DenseTensor<float> Forward(DenseTensor<float> x, ILayerCache cacheBase, int position)
    {
        var cache = (LatentKvCacheLayer)cacheBase;

        float[] ckv = _wDkv.Forward(x).Buffer;                  // [dc]
        float[] kr  = (float[])_wKr.Forward(x).Buffer.Clone();  // [dr]
        RotaryEmbedding.Rotate(kr, 0, _dr, position);
        cache.Append((float[])ckv.Clone(), kr);

        float[] q  = _wQ.Forward(x).Buffer;                     // [numHeads·headDim]
        float[] qr = _wQr.Forward(x).Buffer;                    // [numHeads·dr]
        for (int h = 0; h < _numHeads; h++) RotaryEmbedding.Rotate(qr, h * _dr, _dr, position);

        int n = cache.Length;

        // Reconstruct per-position content K and V from the cached latent.
        var upK = new float[n][];
        var upV = new float[n][];
        for (int p = 0; p < n; p++)
        {
            var cp = new DenseTensor<float>(cache.Ckv(p), _dc);
            upK[p] = _wUk.Forward(cp).Buffer;
            upV[p] = _wUv.Forward(cp).Buffer;
        }

        var ctx  = new DenseTensor<float>(_numHeads * _headDim);
        var ctxB = ctx.Buffer;

        for (int h = 0; h < _numHeads; h++)
        {
            int off  = h * _headDim;
            int roff = h * _dr;
            var scores = new float[n];
            for (int p = 0; p < n; p++)
            {
                float s = LlmMath.Dot(q, off, upK[p], off, _headDim)
                        + LlmMath.Dot(qr, roff, cache.Kr(p), 0, _dr);
                s *= _scale;
                if (_softcap > 0f) s = _softcap * MathF.Tanh(s / _softcap);
                scores[p] = s;
            }
            LlmMath.SoftmaxInPlace(scores, n);
            for (int p = 0; p < n; p++)
            {
                float w = scores[p];
                for (int i = 0; i < _headDim; i++) ctxB[off + i] += w * upV[p][off + i];
            }
        }
        return _wO.Forward(ctx);
    }
}

/// <summary>Builds the attention layer selected by the config.</summary>
public static class Attentions
{
    public static ISelfAttention Create(TransformerConfig cfg, Random rng) => cfg.Attention switch
    {
        AttentionKind.Latent => new LatentAttention(cfg, rng),
        _                    => new MultiHeadAttention(cfg, rng),
    };
}
