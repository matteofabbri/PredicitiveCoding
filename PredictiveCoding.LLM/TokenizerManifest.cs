namespace PredictiveCoding.LLM;

/// <summary>
/// What <see cref="PredictiveLlm.Save"/> records about the tokenizer so that
/// <see cref="PredictiveLlm.Load"/> can reconstruct it and fail LOUDLY on any
/// mismatch (Pro roadmap Phase 0): the kind ("byte", "bpe", or a custom type
/// name), the vocabulary size, and the SHA-256 of the persisted vocab file
/// (empty for stateless tokenizers).
/// </summary>
internal sealed record TokenizerManifest(string Kind, int VocabSize, string Hash);
