namespace PredictiveCoding.LLM.Autograd;

// ─────────────────────────────────────────────────────────────────────────────
// DecoderLM  —  a configurable multi-block, multi-head decoder transformer built
// on the autograd engine (the trainable counterpart of the inference-only
// GptModel). It generalises TinyTransformerLM from a single single-head block to
// N blocks × H heads, and adds sinusoidal positional encoding + a final norm.
//
// Multi-head without a concat op: each head h owns its own Q/K/V projections into
// head-space AND its own output projection Wo_h [headDim, dim]; the per-head
// outputs are summed. That sum is exactly  concat(head_1..head_H) · Wo  with a
// block-structured Wo, so it is real multi-head attention expressed only with the
// ops the engine already has (MatMul, Transpose, Scale, SoftmaxRowsCausal,
// LayerNorm, Gelu, Add, AddBias, Dropout).
//
// Forward (pre-norm + residual, GPT-style):
//   x  = onehot(tokens)·Wemb + positional
//   for each block:
//     x += Dropout( Σ_h  softmaxCausal((h1·Wq_h)(h1·Wk_h)ᵀ·scale)·(h1·Wv_h) · Wo_h )
//     x += Dropout( gelu(LN(x)·W1 + b1)·W2 + b2 )
//   logits = LN_f(x)·Wout
// ─────────────────────────────────────────────────────────────────────────────

public sealed record DecoderLmConfig(
    int Vocab,
    int Dim,
    int NumBlocks,
    int NumHeads,
    int FfDim,
    int MaxSeq,
    float Dropout = 0f,
    float Eps = 1e-5f)
{
    public int HeadDim => Dim / NumHeads;

    public void Validate()
    {
        if (Vocab <= 0 || Dim <= 0 || NumBlocks <= 0 || NumHeads <= 0 || FfDim <= 0 || MaxSeq <= 0)
            throw new ArgumentException("All DecoderLmConfig sizes must be positive.");
        if (Dim % NumHeads != 0)
            throw new ArgumentException($"Dim ({Dim}) must be divisible by NumHeads ({NumHeads}).");
    }
}

public sealed class DecoderLM
{
    public DecoderLmConfig Config { get; }

    private readonly float    _attnScale;
    private readonly Random   _rng;
    private readonly Tensor   _wemb;          // [vocab, dim]
    private readonly Block[]  _blocks;
    private readonly Tensor   _lnFG, _lnFB;   // final layer-norm
    private readonly Tensor   _wout;          // [dim, vocab]

    public DecoderLM(DecoderLmConfig cfg, Random rng)
    {
        cfg.Validate();
        Config     = cfg;
        _rng       = rng;
        _attnScale = 1f / MathF.Sqrt(cfg.HeadDim);

        _wemb   = AgInit.Randn(cfg.Vocab, cfg.Dim, rng);
        _blocks = new Block[cfg.NumBlocks];
        for (int b = 0; b < cfg.NumBlocks; b++) _blocks[b] = new Block(cfg, rng);
        _lnFG = AgInit.Ones(1, cfg.Dim);
        _lnFB = new Tensor(1, cfg.Dim);
        _wout = AgInit.Randn(cfg.Dim, cfg.Vocab, rng);
    }

    /// <summary>All trainable tensors in a fixed order (must match for Checkpoint save/load).</summary>
    public Tensor[] Parameters()
    {
        var ps = new List<Tensor> { _wemb };
        foreach (var b in _blocks) ps.AddRange(b.Parameters());
        ps.Add(_lnFG);
        ps.Add(_lnFB);
        ps.Add(_wout);
        return [.. ps];
    }

    /// <summary>Forward pass: tokens → logits [seq, vocab]. Dropout active only when training.</summary>
    public Tensor Forward(int[] tokens, bool training = false)
    {
        int seq = tokens.Length;
        if (seq > Config.MaxSeq)
            throw new ArgumentException($"Sequence length {seq} exceeds MaxSeq {Config.MaxSeq}.");

        var oneHot = new Tensor(seq, Config.Vocab);
        for (int i = 0; i < seq; i++) oneHot.Data[i * Config.Vocab + tokens[i]] = 1f;

        var x = Ag.MatMul(oneHot, _wemb);                                  // [seq, dim]
        x = Ag.Add(x, AgPositional.Sinusoidal(seq, Config.Dim));           // + positions

        foreach (var blk in _blocks)
            x = blk.Forward(x, _attnScale, Config.Dropout, training, _rng, Config.Eps);

        var xf = Ag.LayerNorm(x, _lnFG, _lnFB, Config.Eps);
        return Ag.MatMul(xf, _wout);                                       // [seq, vocab]
    }

    /// <summary>Masked next-token cross-entropy; rows with target == ignoreIndex are skipped.</summary>
    public Tensor Loss(int[] tokens, int[] targets, int ignoreIndex, bool training = false)
        => Ag.CrossEntropy(Forward(tokens, training), targets, ignoreIndex);

    // ── one transformer block ────────────────────────────────────────────────────

    private sealed class Block
    {
        private readonly int      _heads;
        private readonly Tensor   _ln1G, _ln1B, _ln2G, _ln2B;
        private readonly Tensor[] _wq, _wk, _wv, _wo;          // per-head projections
        private readonly Tensor   _w1, _b1, _w2, _b2;          // feed-forward

        public Block(DecoderLmConfig cfg, Random rng)
        {
            _heads = cfg.NumHeads;
            _ln1G = AgInit.Ones(1, cfg.Dim); _ln1B = new Tensor(1, cfg.Dim);
            _ln2G = AgInit.Ones(1, cfg.Dim); _ln2B = new Tensor(1, cfg.Dim);

            _wq = new Tensor[_heads]; _wk = new Tensor[_heads];
            _wv = new Tensor[_heads]; _wo = new Tensor[_heads];
            for (int h = 0; h < _heads; h++)
            {
                _wq[h] = AgInit.Randn(cfg.Dim, cfg.HeadDim, rng);
                _wk[h] = AgInit.Randn(cfg.Dim, cfg.HeadDim, rng);
                _wv[h] = AgInit.Randn(cfg.Dim, cfg.HeadDim, rng);
                _wo[h] = AgInit.Randn(cfg.HeadDim, cfg.Dim, rng);
            }

            _w1 = AgInit.Randn(cfg.Dim, cfg.FfDim, rng); _b1 = new Tensor(1, cfg.FfDim);
            _w2 = AgInit.Randn(cfg.FfDim, cfg.Dim, rng); _b2 = new Tensor(1, cfg.Dim);
        }

        public IEnumerable<Tensor> Parameters()
        {
            yield return _ln1G; yield return _ln1B;
            for (int h = 0; h < _heads; h++)
            {
                yield return _wq[h]; yield return _wk[h]; yield return _wv[h]; yield return _wo[h];
            }
            yield return _ln2G; yield return _ln2B;
            yield return _w1; yield return _b1; yield return _w2; yield return _b2;
        }

        public Tensor Forward(Tensor x, float attnScale, float dropout, bool training, Random rng, float eps)
        {
            // Multi-head causal self-attention (pre-norm + residual).
            var h1 = Ag.LayerNorm(x, _ln1G, _ln1B, eps);
            Tensor? attn = null;
            for (int h = 0; h < _heads; h++)
            {
                var q       = Ag.MatMul(h1, _wq[h]);                                  // [seq, headDim]
                var k       = Ag.MatMul(h1, _wk[h]);
                var v       = Ag.MatMul(h1, _wv[h]);
                var scores  = Ag.Scale(Ag.MatMul(q, Ag.Transpose(k)), attnScale);     // [seq, seq]
                var headOut = Ag.MatMul(Ag.SoftmaxRowsCausal(scores), v);             // [seq, headDim]
                var proj    = Ag.MatMul(headOut, _wo[h]);                             // [seq, dim]
                attn = attn is null ? proj : Ag.Add(attn, proj);
            }
            x = Ag.Add(x, Ag.Dropout(attn!, dropout, training, rng));

            // Feed-forward (pre-norm + residual).
            var h2  = Ag.LayerNorm(x, _ln2G, _ln2B, eps);
            var ff1 = Ag.Gelu(Ag.AddBias(Ag.MatMul(h2, _w1), _b1));
            var ff2 = Ag.Dropout(Ag.AddBias(Ag.MatMul(ff1, _w2), _b2), dropout, training, rng);
            return Ag.Add(x, ff2);
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AgInit  —  shared parameter initialisers for autograd models.
// ─────────────────────────────────────────────────────────────────────────────

internal static class AgInit
{
    public static Tensor Randn(int rows, int cols, Random rng)
    {
        var t = new Tensor(rows, cols);
        for (int i = 0; i < t.Data.Length; i++) t.Data[i] = 0.02f * (float)NextGaussian(rng);
        return t;
    }

    public static Tensor Ones(int rows, int cols)
    {
        var t = new Tensor(rows, cols);
        Array.Fill(t.Data, 1f);
        return t;
    }

    private static double NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
