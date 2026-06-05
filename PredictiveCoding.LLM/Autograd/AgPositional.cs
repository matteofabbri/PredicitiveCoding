namespace PredictiveCoding.LLM.Autograd;

// ─────────────────────────────────────────────────────────────────────────────
// AgPositional  —  positional encoding for the autograd (trainable) path.
//
// TinyTransformerLM / DecoderLM add positions to the token embeddings so the
// attention is sensitive to word order. Without them a decoder block is
// permutation-equivariant apart from the causal mask — fine for a toy LM, fatal
// for translation, where word order carries meaning. These encodings are constant
// (no learnable parameters); add them to the embeddings with Ag.Add.
// ─────────────────────────────────────────────────────────────────────────────

public static class AgPositional
{
    /// <summary>
    /// Fixed sinusoidal positional encoding (Vaswani et al. 2017) as a constant
    /// [seq, dim] tensor: PE[pos, 2i] = sin(pos / 10000^(2i/dim)),
    /// PE[pos, 2i+1] = cos(pos / 10000^(2i/dim)).
    /// </summary>
    public static Tensor Sinusoidal(int seq, int dim)
    {
        var t = new Tensor(seq, dim);
        for (int pos = 0; pos < seq; pos++)
            for (int i = 0; i < dim; i++)
            {
                float denom = MathF.Pow(10000f, (2f * (i / 2)) / dim);
                float angle = pos / denom;
                t.Data[pos * dim + i] = (i % 2 == 0) ? MathF.Sin(angle) : MathF.Cos(angle);
            }
        return t;
    }
}
