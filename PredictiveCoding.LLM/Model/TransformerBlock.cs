using PredictiveCoding.Core;

namespace PredictiveCoding.LLM.Model;

// ─────────────────────────────────────────────────────────────────────────────
// FeedForward variants + TransformerBlock  —  the repeating unit of the decoder.
//
// Pre-norm residual:
//   x ← x + Attention(Norm(x))
//   x ← x + FeedForward(Norm(x))
// ─────────────────────────────────────────────────────────────────────────────

public interface IFeedForward
{
    DenseTensor<float> Forward(DenseTensor<float> x);
}

/// <summary>Position-wise GELU MLP (GPT-2): Linear → GELU → Linear.</summary>
public sealed class FeedForward : IFeedForward
{
    private readonly Linear _fc1, _fc2;

    public FeedForward(TransformerConfig cfg, Random rng)
    {
        _fc1 = new Linear(cfg.EmbedDim, cfg.FeedForwardDim, rng, ActivationType.ReLU);
        _fc2 = new Linear(cfg.FeedForwardDim, cfg.EmbedDim, rng);
    }

    public DenseTensor<float> Forward(DenseTensor<float> x)
    {
        var h = _fc1.Forward(x);
        LlmMath.GeluInPlace(h.Buffer);
        return _fc2.Forward(h);
    }
}

/// <summary>Gated SwiGLU FFN (Shazeer 2020): down( SiLU(gate(x)) ⊙ up(x) ).</summary>
public sealed class SwiGluFeedForward : IFeedForward
{
    private readonly Linear _gate, _up, _down;

    public SwiGluFeedForward(TransformerConfig cfg, Random rng)
    {
        _gate = new Linear(cfg.EmbedDim, cfg.FeedForwardDim, rng, ActivationType.ReLU);
        _up   = new Linear(cfg.EmbedDim, cfg.FeedForwardDim, rng);
        _down = new Linear(cfg.FeedForwardDim, cfg.EmbedDim, rng);
    }

    public DenseTensor<float> Forward(DenseTensor<float> x)
    {
        var gate = _gate.Forward(x);
        var up   = _up.Forward(x);
        LlmMath.SiluInPlace(gate.Buffer);
        return _down.Forward(TensorUtils.ElementwiseMul(gate, up));
    }
}

public static class FeedForwards
{
    public static IFeedForward Create(TransformerConfig cfg, Random rng) => cfg.Ffn switch
    {
        FeedForwardKind.SwiGlu => new SwiGluFeedForward(cfg, rng),
        _                      => new FeedForward(cfg, rng),
    };
}

public sealed class TransformerBlock
{
    private readonly INormLayer     _ln1, _ln2;
    private readonly ISelfAttention _attn;
    private readonly IFeedForward   _ffn;

    public TransformerBlock(TransformerConfig cfg, Random rng)
    {
        _ln1  = Norms.Create(cfg);
        _ln2  = Norms.Create(cfg);
        _attn = Attentions.Create(cfg, rng);
        _ffn  = FeedForwards.Create(cfg, rng);
    }

    public ILayerCache CreateCache() => _attn.CreateCache();

    public DenseTensor<float> Forward(DenseTensor<float> x, ILayerCache cache, int position)
    {
        var a = _attn.Forward(_ln1.Forward(x), cache, position);
        var h = x.Clone();
        TensorUtils.InPlaceAxpy(1f, a, h);

        var f = _ffn.Forward(_ln2.Forward(h));
        TensorUtils.InPlaceAxpy(1f, f, h);
        return h;
    }
}
