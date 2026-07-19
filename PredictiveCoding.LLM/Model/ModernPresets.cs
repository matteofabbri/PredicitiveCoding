namespace PredictiveCoding.LLM.Model;

// ─────────────────────────────────────────────────────────────────────────────
// ModernPresets  —  configs assembled from recent (2024-2025) LLM papers.
//
//   • Llama3Style : RMSNorm + SwiGLU + RoPE + grouped-query attention
//     (Llama 3 / Qwen2-3 / Mistral).
//   • NemotronStyle : the above with a 3:1 temporal:attention hybrid interleave
//     (Nemotron 3 / Mamba-hybrid, 2026) — O(1) cache on the temporal layers.
//   • DeepSeekStyle : the above, plus Multi-head Latent Attention (DeepSeek-V2/V3),
//     QK-Norm (OLMo 2 / Gemma 2 / Qwen3) and attention/final logit soft-capping
//     (Gemma 2).
//
// References:
//   DeepSeek-V3 Technical Report (arXiv:2412.19437) — MLA, DeepSeekMoE, MTP.
//   Shazeer (2020) "GLU Variants Improve Transformer" — SwiGLU.
//   Zhang & Sennrich (2019) — RMSNorm.   Su et al. (2021) — RoPE.
//   Gemma 2 (2024) — logit soft-capping.   OLMo 2 (2024) — QK-Norm.
// ─────────────────────────────────────────────────────────────────────────────

public static class ModernPresets
{
    /// <summary>Llama-3 / Qwen-3 style: RMSNorm + SwiGLU + RoPE + GQA.</summary>
    public static TransformerConfig Llama3Style(
        int vocab = 260, int maxSeq = 256, int embed = 128, int heads = 4,
        int kvHeads = 2, int layers = 4, int ff = 512)
        => new()
        {
            VocabSize = vocab, MaxSeqLen = maxSeq, EmbedDim = embed, NumHeads = heads,
            NumKvHeads = kvHeads, NumLayers = layers, FeedForwardDim = ff,
            Norm = NormKind.RmsNorm, Ffn = FeedForwardKind.SwiGlu, Positional = PositionalKind.Rotary,
        };

    /// <summary>
    /// Nemotron-3-style hybrid (2026): attention interleaved with gated
    /// temporal-state layers — 3 temporal : 1 attention, O(1) cache on the
    /// temporal layers — plus RMSNorm + SwiGLU + RoPE + GQA.
    /// </summary>
    public static TransformerConfig NemotronStyle(
        int vocab = 260, int maxSeq = 256, int embed = 128, int heads = 4,
        int kvHeads = 2, int layers = 4, int ff = 512)
    {
        // 3:1 temporal:attention interleave, attention every 4th layer.
        var pattern = new char[layers];
        for (int l = 0; l < layers; l++) pattern[l] = (l + 1) % 4 == 0 ? 'A' : 'T';

        return new TransformerConfig
        {
            VocabSize = vocab, MaxSeqLen = maxSeq, EmbedDim = embed, NumHeads = heads,
            NumKvHeads = kvHeads, NumLayers = layers, FeedForwardDim = ff,
            Norm = NormKind.RmsNorm, Ffn = FeedForwardKind.SwiGlu, Positional = PositionalKind.Rotary,
            BlockPattern = new string(pattern),
        };
    }

    /// <summary>DeepSeek-V3 style: MLA + SwiGLU + RMSNorm + QK-Norm + Gemma-2 soft-caps.</summary>
    public static TransformerConfig DeepSeekStyle(
        int vocab = 260, int maxSeq = 256, int embed = 128, int heads = 4,
        int layers = 4, int ff = 512)
    {
        int headDim = embed / heads;
        int rope    = Math.Max(2, headDim / 2);
        if (rope % 2 != 0) rope--;

        return new TransformerConfig
        {
            VocabSize = vocab, MaxSeqLen = maxSeq, EmbedDim = embed, NumHeads = heads,
            NumLayers = layers, FeedForwardDim = ff,
            Norm = NormKind.RmsNorm, Ffn = FeedForwardKind.SwiGlu, Positional = PositionalKind.Rotary,
            UseQkNorm = true, AttnLogitSoftcap = 50f, FinalLogitSoftcap = 30f,
            Attention = AttentionKind.Latent, KvLatentDim = Math.Max(1, embed / 2), RopeHeadDim = rope,
        };
    }
}
