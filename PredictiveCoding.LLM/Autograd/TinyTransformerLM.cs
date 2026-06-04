namespace PredictiveCoding.LLM.Autograd;

// ─────────────────────────────────────────────────────────────────────────────
// TinyTransformerLM  —  a single-block decoder transformer on the autograd engine.
//
// Purpose: demonstrate (and gradient-check) full backpropagation through a
// transformer — token embedding → LayerNorm → causal self-attention → residual →
// LayerNorm → GELU feed-forward → residual → LM head → cross-entropy. It is the
// trainable counterpart of the fast (inference-only) GptModel.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class TinyTransformerLM
{
    private readonly int   _vocab;
    private readonly int   _dim;
    private readonly float _eps;
    private readonly float _attnScale;

    // Parameters.
    public Tensor Wemb { get; }                       // [vocab, dim]
    public Tensor Wq, Wk, Wv, Wo;                      // [dim, dim]
    public Tensor Ln1G, Ln1B, Ln2G, Ln2B;             // [1, dim]
    public Tensor W1, B1;                              // [dim, ff], [1, ff]
    public Tensor W2, B2;                              // [ff, dim], [1, dim]
    public Tensor Wout;                                // [dim, vocab]

    public TinyTransformerLM(int vocab, int dim, int ff, Random rng, float eps = 1e-5f)
    {
        _vocab = vocab;
        _dim   = dim;
        _eps   = eps;
        _attnScale = 1f / MathF.Sqrt(dim);

        Wemb = Randn(vocab, dim, rng);
        Wq   = Randn(dim, dim, rng); Wk = Randn(dim, dim, rng);
        Wv   = Randn(dim, dim, rng); Wo = Randn(dim, dim, rng);
        Ln1G = Ones(1, dim); Ln1B = new Tensor(1, dim);
        Ln2G = Ones(1, dim); Ln2B = new Tensor(1, dim);
        W1 = Randn(dim, ff, rng); B1 = new Tensor(1, ff);
        W2 = Randn(ff, dim, rng); B2 = new Tensor(1, dim);
        Wout = Randn(dim, vocab, rng);
    }

    public Tensor[] Parameters() =>
        [Wemb, Wq, Wk, Wv, Wo, Ln1G, Ln1B, Ln2G, Ln2B, W1, B1, W2, B2, Wout];

    /// <summary>Forward pass: tokens → logits [seq, vocab].</summary>
    public Tensor Forward(int[] tokens)
    {
        int seq = tokens.Length;

        // One-hot input × embedding table (constant input, trainable table).
        var oneHot = new Tensor(seq, _vocab);
        for (int i = 0; i < seq; i++) oneHot.Data[i * _vocab + tokens[i]] = 1f;
        var x = Ag.MatMul(oneHot, Wemb);                              // [seq, dim]

        // Causal self-attention sub-layer (pre-norm + residual).
        var h1     = Ag.LayerNorm(x, Ln1G, Ln1B, _eps);
        var q      = Ag.MatMul(h1, Wq);
        var k      = Ag.MatMul(h1, Wk);
        var v      = Ag.MatMul(h1, Wv);
        var scores = Ag.Scale(Ag.MatMul(q, Ag.Transpose(k)), _attnScale);   // [seq, seq]
        var attn   = Ag.MatMul(Ag.SoftmaxRowsCausal(scores), v);            // [seq, dim]
        x = Ag.Add(x, Ag.MatMul(attn, Wo));

        // Feed-forward sub-layer (pre-norm + residual).
        var h2  = Ag.LayerNorm(x, Ln2G, Ln2B, _eps);
        var ff1 = Ag.Gelu(Ag.AddBias(Ag.MatMul(h2, W1), B1));
        var ff2 = Ag.AddBias(Ag.MatMul(ff1, W2), B2);
        x = Ag.Add(x, ff2);

        return Ag.MatMul(x, Wout);                                   // [seq, vocab]
    }

    /// <summary>Cross-entropy next-token loss for (tokens, targets).</summary>
    public Tensor Loss(int[] tokens, int[] targets) => Ag.CrossEntropy(Forward(tokens), targets);

    // ── parameter init ───────────────────────────────────────────────────────

    private static Tensor Randn(int rows, int cols, Random rng)
    {
        var t = new Tensor(rows, cols);
        for (int i = 0; i < t.Data.Length; i++)
            t.Data[i] = 0.02f * (float)NextGaussian(rng);
        return t;
    }

    private static Tensor Ones(int rows, int cols)
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

// ─────────────────────────────────────────────────────────────────────────────
// Sgd  —  plain stochastic gradient descent step (the autograd training demo).
// A proper optimizer (Adam / AdamW) is a separate roadmap item.
// ─────────────────────────────────────────────────────────────────────────────

public static class Sgd
{
    public static void Step(IEnumerable<Tensor> parameters, float learningRate)
    {
        foreach (var p in parameters)
            for (int i = 0; i < p.Data.Length; i++)
                p.Data[i] -= learningRate * p.Grad[i];
    }
}
