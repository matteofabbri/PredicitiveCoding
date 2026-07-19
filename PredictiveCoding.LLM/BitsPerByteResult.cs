namespace PredictiveCoding.LLM;

/// <summary>
/// Tokenizer-independent yardstick (Pro roadmap Phase 0): with a subword
/// vocabulary, perplexity-per-token is incomparable across tokenizers — the
/// honest metric is how many bits the model spends per BYTE of text.
/// Computed over non-overlapping windows; bytes are counted on the decoded
/// scored targets, so byte and BPE models are measured on the same footing.
/// </summary>
/// <param name="BitsPerByte">Total NLL in bits ÷ scored text bytes — lower is better (8 = incompressible).</param>
/// <param name="Bytes">UTF-8 bytes actually scored.</param>
/// <param name="Tokens">Tokens actually scored.</param>
public sealed record BitsPerByteResult(double BitsPerByte, long Bytes, long Tokens)
{
    public override string ToString() => $"{BitsPerByte:F3} bits/byte over {Bytes} bytes ({Tokens} tokens)";
}
