using PredictiveCoding.Core;

namespace PredictiveCoding.LLM.Model;

// ─────────────────────────────────────────────────────────────────────────────
// FeedForward + TransformerBlock  —  the repeating unit of the decoder.
//
// Pre-norm residual formulation (GPT-2 style):
//   x ← x + Attention(LayerNorm(x))
//   x ← x + FeedForward(LayerNorm(x))
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Position-wise feed-forward network: Linear → GELU → Linear.</summary>
public sealed class FeedForward
{
    private readonly Linear _fc1;
    private readonly Linear _fc2;

    public FeedForward(TransformerConfig cfg, Random rng)
    {
        _fc1 = new Linear(cfg.EmbedDim, cfg.FeedForwardDim, rng, ActivationType.ReLU); // He init for GELU/ReLU family
        _fc2 = new Linear(cfg.FeedForwardDim, cfg.EmbedDim, rng);
    }

    public DenseTensor<float> Forward(DenseTensor<float> x)
    {
        var h = _fc1.Forward(x);
        LlmMath.GeluInPlace(h.Buffer);
        return _fc2.Forward(h);
    }
}

public sealed class TransformerBlock
{
    private readonly LayerNorm            _ln1;
    private readonly LayerNorm            _ln2;
    private readonly MultiHeadAttention   _attn;
    private readonly FeedForward          _ffn;

    public TransformerBlock(TransformerConfig cfg, Random rng)
    {
        _ln1  = new LayerNorm(cfg.EmbedDim, cfg.LayerNormEps);
        _ln2  = new LayerNorm(cfg.EmbedDim, cfg.LayerNormEps);
        _attn = new MultiHeadAttention(cfg, rng);
        _ffn  = new FeedForward(cfg, rng);
    }

    public DenseTensor<float> Forward(DenseTensor<float> x, KvCacheLayer cache)
    {
        // x = x + Attention(LN(x))
        var a = _attn.Forward(_ln1.Forward(x), cache);
        var h = x.Clone();
        TensorUtils.InPlaceAxpy(1f, a, h);

        // h = h + FFN(LN(h))
        var f = _ffn.Forward(_ln2.Forward(h));
        TensorUtils.InPlaceAxpy(1f, f, h);
        return h;
    }
}
