namespace PredictiveCoding.LLM.Model;

/// <summary>Positional information scheme.</summary>
public enum PositionalKind
{
    /// <summary>Additive sinusoidal encoding added to the token embedding.</summary>
    Sinusoidal,
    /// <summary>Rotary position embedding (RoPE) applied to Q/K inside attention.</summary>
    Rotary,
    /// <summary>ALiBi: a per-head linear distance bias added to the attention scores.</summary>
    Alibi,
}

/// <summary>Normalization layer kind.</summary>
public enum NormKind
{
    LayerNorm,
    RmsNorm,
}

// ─────────────────────────────────────────────────────────────────────────────
// TransformerConfig  —  hyperparameters for the decoder-only transformer (GPT).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record TransformerConfig
{
    /// <summary>Vocabulary size (must match the tokenizer).</summary>
    public int VocabSize { get; init; } = 260;

    /// <summary>Maximum context length (number of positions).</summary>
    public int MaxSeqLen { get; init; } = 256;

    /// <summary>Model / embedding dimensionality (d_model).</summary>
    public int EmbedDim { get; init; } = 128;

    /// <summary>Number of attention (query) heads (must divide EmbedDim).</summary>
    public int NumHeads { get; init; } = 4;

    /// <summary>
    /// Number of key/value heads (grouped-query attention). Must divide NumHeads.
    /// Equal to NumHeads → standard MHA; 1 → multi-query attention (MQA);
    /// in between → grouped-query attention (GQA). Default: same as NumHeads.
    /// </summary>
    public int NumKvHeads { get; init; } = 0;   // 0 ⇒ resolve to NumHeads

    /// <summary>Number of stacked transformer blocks.</summary>
    public int NumLayers { get; init; } = 4;

    /// <summary>Inner dimensionality of the feed-forward block (usually 4·EmbedDim).</summary>
    public int FeedForwardDim { get; init; } = 512;

    /// <summary>Epsilon for normalisation.</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>Tie the input embedding and the output (LM head) projection.</summary>
    public bool TieEmbeddings { get; init; } = true;

    /// <summary>Positional scheme (sinusoidal or rotary/RoPE).</summary>
    public PositionalKind Positional { get; init; } = PositionalKind.Sinusoidal;

    /// <summary>Normalisation kind (LayerNorm or RMSNorm).</summary>
    public NormKind Norm { get; init; } = NormKind.LayerNorm;

    /// <summary>
    /// Use the flash-attention-style online-softmax accumulation (single pass, no
    /// full score vector). Numerically equivalent to the standard path.
    /// </summary>
    public bool UseFlashAttention { get; init; } = false;

    /// <summary>Per-head dimensionality = EmbedDim / NumHeads.</summary>
    public int HeadDim => EmbedDim / NumHeads;

    /// <summary>Resolved number of key/value heads.</summary>
    public int KvHeads => NumKvHeads <= 0 ? NumHeads : NumKvHeads;

    /// <summary>How many query heads share one key/value head.</summary>
    public int KvGroupSize => NumHeads / KvHeads;

    public void Validate()
    {
        if (VocabSize < 2)       throw new InvalidOperationException("VocabSize must be >= 2.");
        if (MaxSeqLen < 1)       throw new InvalidOperationException("MaxSeqLen must be >= 1.");
        if (EmbedDim < 1)        throw new InvalidOperationException("EmbedDim must be >= 1.");
        if (NumHeads < 1)        throw new InvalidOperationException("NumHeads must be >= 1.");
        if (EmbedDim % NumHeads != 0)
            throw new InvalidOperationException($"EmbedDim ({EmbedDim}) must be divisible by NumHeads ({NumHeads}).");
        if (KvHeads < 1 || NumHeads % KvHeads != 0)
            throw new InvalidOperationException($"NumHeads ({NumHeads}) must be divisible by KvHeads ({KvHeads}).");
        if (Positional == PositionalKind.Rotary && HeadDim % 2 != 0)
            throw new InvalidOperationException("RoPE requires an even HeadDim.");
        if (NumLayers < 1)       throw new InvalidOperationException("NumLayers must be >= 1.");
        if (FeedForwardDim < 1)  throw new InvalidOperationException("FeedForwardDim must be >= 1.");
        if (LayerNormEps <= 0f)  throw new InvalidOperationException("LayerNormEps must be > 0.");
    }
}
