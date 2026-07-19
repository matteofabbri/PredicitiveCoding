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

    internal Linear Fc1 => _fc1;   // serving-path import hooks (ROADMAP-Pro Phase 1)
    internal Linear Fc2 => _fc2;

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

/// <summary>
/// Inference-side sparse mixture of experts (ROADMAP-Pro Phase 1): softmax
/// router → hard top-k → gate-weighted expert sum, unrenormalised — the exact
/// semantics of the trainable MoE, so exported weights behave identically.
/// Genuinely FLOP-sparse here for free: only the selected experts run.
/// </summary>
public sealed class MoeFeedForward : IFeedForward
{
    private readonly Linear       _router;    // [E, dim]
    private readonly FeedForward[] _experts;
    private readonly int          _topK;

    internal Linear        Router  => _router;    // serving-path import hooks
    internal FeedForward[] Experts => _experts;

    public MoeFeedForward(TransformerConfig cfg, Random rng)
    {
        _router  = new Linear(cfg.EmbedDim, cfg.NumExperts, rng);
        _experts = new FeedForward[cfg.NumExperts];
        for (int e = 0; e < cfg.NumExperts; e++) _experts[e] = new FeedForward(cfg, rng);
        _topK = cfg.MoeTopK;
    }

    public DenseTensor<float> Forward(DenseTensor<float> x)
    {
        float[] gates = (float[])_router.Forward(x).Buffer.Clone();
        LlmMath.SoftmaxInPlace(gates, gates.Length);

        var outT = new DenseTensor<float>(x.Length);
        for (int chosen = 0; chosen < _topK; chosen++)
        {
            int best = -1;
            float bestGate = float.NegativeInfinity;
            for (int e = 0; e < gates.Length; e++)
                if (gates[e] > bestGate) { bestGate = gates[e]; best = e; }
            gates[best] = float.NegativeInfinity;                  // taken

            var y = _experts[best].Forward(x);
            TensorUtils.InPlaceAxpy(bestGate, y, outT);            // out += gate · expert(x)
        }
        return outT;
    }
}

public static class FeedForwards
{
    public static IFeedForward Create(TransformerConfig cfg, Random rng)
        => cfg.NumExperts > 0 ? new MoeFeedForward(cfg, rng)
         : cfg.Ffn switch
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

    internal INormLayer     Ln1  => _ln1;   // serving-path import hooks
    internal INormLayer     Ln2  => _ln2;
    internal ISelfAttention Attn => _attn;
    internal IFeedForward   Ffn  => _ffn;

    public TransformerBlock(TransformerConfig cfg, Random rng, bool temporal = false)
    {
        _ln1  = Norms.Create(cfg);
        _ln2  = Norms.Create(cfg);
        _attn = temporal ? new RecurrentTemporalLayer(cfg, rng) : Attentions.Create(cfg, rng);
        // MoE lives in the attention blocks; temporal blocks keep a dense FFN
        // (mirrors the trainable stack, so weight export maps 1:1).
        _ffn  = temporal && cfg.NumExperts > 0 ? new FeedForward(cfg, rng) : FeedForwards.Create(cfg, rng);
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
