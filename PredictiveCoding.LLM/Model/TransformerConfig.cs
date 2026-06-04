namespace PredictiveCoding.LLM.Model;

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

    /// <summary>Number of attention heads (must divide EmbedDim).</summary>
    public int NumHeads { get; init; } = 4;

    /// <summary>Number of stacked transformer blocks.</summary>
    public int NumLayers { get; init; } = 4;

    /// <summary>Inner dimensionality of the feed-forward block (usually 4·EmbedDim).</summary>
    public int FeedForwardDim { get; init; } = 512;

    /// <summary>Epsilon for layer normalisation.</summary>
    public float LayerNormEps { get; init; } = 1e-5f;

    /// <summary>Tie the input embedding and the output (LM head) projection.</summary>
    public bool TieEmbeddings { get; init; } = true;

    /// <summary>Per-head dimensionality = EmbedDim / NumHeads.</summary>
    public int HeadDim => EmbedDim / NumHeads;

    public void Validate()
    {
        if (VocabSize < 2)       throw new InvalidOperationException("VocabSize must be >= 2.");
        if (MaxSeqLen < 1)       throw new InvalidOperationException("MaxSeqLen must be >= 1.");
        if (EmbedDim < 1)        throw new InvalidOperationException("EmbedDim must be >= 1.");
        if (NumHeads < 1)        throw new InvalidOperationException("NumHeads must be >= 1.");
        if (EmbedDim % NumHeads != 0)
            throw new InvalidOperationException($"EmbedDim ({EmbedDim}) must be divisible by NumHeads ({NumHeads}).");
        if (NumLayers < 1)       throw new InvalidOperationException("NumLayers must be >= 1.");
        if (FeedForwardDim < 1)  throw new InvalidOperationException("FeedForwardDim must be >= 1.");
        if (LayerNormEps <= 0f)  throw new InvalidOperationException("LayerNormEps must be > 0.");
    }
}
