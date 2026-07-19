using PredictiveCoding.Core;
using PredictiveCoding.LLM.Eval;

namespace PredictiveCoding.LLM.Model;

// ─────────────────────────────────────────────────────────────────────────────
// GptModel  —  decoder-only transformer language model.
//
//   token → embedding (+ positional) → N transformer blocks → final LayerNorm
//         → LM head (tied to the embedding, or a separate Linear) → vocab logits
//
// The model is stepped one token at a time, threading a KvCache through the
// blocks so generation is incremental.  Weights are randomly initialised; the
// surrounding PredictiveCoding library is backprop-free, so a gradient trainer is
// a separate (future) component — see LLM_Roadmap.md.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class GptModel
{
    private readonly TransformerConfig   _cfg;
    private readonly Embedding           _embedding;
    private readonly PositionalEncoding? _positional;   // null when using RoPE
    private readonly TransformerBlock[]  _blocks;
    private readonly INormLayer          _lnFinal;
    private readonly Linear?             _lmHead;   // null when embeddings are tied

    public TransformerConfig Configuration => _cfg;
    public int VocabSize => _cfg.VocabSize;
    public int MaxSeqLen => _cfg.MaxSeqLen;

    public GptModel(TransformerConfig cfg, Random rng)
    {
        cfg.Validate();
        _cfg        = cfg;
        _embedding  = new Embedding(cfg.VocabSize, cfg.EmbedDim, rng);
        _positional = cfg.Positional == PositionalKind.Sinusoidal ? new PositionalEncoding(cfg.EmbedDim) : null;
        _blocks     = new TransformerBlock[cfg.NumLayers];
        for (int l = 0; l < cfg.NumLayers; l++)
            _blocks[l] = new TransformerBlock(cfg, rng, temporal: cfg.BlockPattern?[l] == 'T');
        _lnFinal    = Norms.Create(cfg);
        _lmHead     = cfg.TieEmbeddings ? null : new Linear(cfg.EmbedDim, cfg.VocabSize, rng);
    }

    /// <summary>Allocates a fresh key/value cache for one generation/sequence.</summary>
    public KvCache CreateCache()
    {
        var layers = new ILayerCache[_blocks.Length];
        for (int l = 0; l < _blocks.Length; l++) layers[l] = _blocks[l].CreateCache();
        return new KvCache(layers);
    }

    /// <summary>
    /// Runs one decoding step: embeds <paramref name="token"/> at
    /// <paramref name="position"/>, updates the cache, and returns the next-token
    /// logits (length VocabSize). An optional <see cref="ActivationProbe"/>
    /// records per-block hidden-state statistics; null (the default) leaves the
    /// step untouched.
    /// </summary>
    public DenseTensor<float> Step(int token, int position, KvCache cache, ActivationProbe? probe = null)
    {
        var x = _embedding.Lookup(token);
        _positional?.AddTo(x, position);

        for (int l = 0; l < _blocks.Length; l++)
        {
            x = _blocks[l].Forward(x, cache.Layers[l], position);
            probe?.RecordBlock(l, x.Buffer);
        }

        x = _lnFinal.Forward(x);
        var logits = _lmHead is null ? _embedding.Project(x) : _lmHead.Forward(x);

        if (_cfg.FinalLogitSoftcap > 0f)   // Gemma 2 final-logit soft-capping
        {
            float cap = _cfg.FinalLogitSoftcap;
            var b = logits.Buffer;
            for (int i = 0; i < b.Length; i++) b[i] = cap * MathF.Tanh(b[i] / cap);
        }
        return logits;
    }
}
