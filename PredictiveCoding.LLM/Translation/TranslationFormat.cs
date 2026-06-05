using PredictiveCoding.LLM.Tokenization;

namespace PredictiveCoding.LLM.Translation;

// ─────────────────────────────────────────────────────────────────────────────
// TranslationFormat  —  turns a (source, target, target-language) triple into the
// single token sequence a decoder-only LM trains on, plus the target-side loss
// mask.
//
// Layout:   [BOS] >>tgt<< {source}\n {target} [EOS]
//
//   • ">>tgt<<" is a plain-text target-language tag (e.g. >>ita<<). It needs no
//     special token — the byte-level BPE encodes it like any text — and lets one
//     model translate *any* input language into a chosen target (many-to-target).
//   • The newline separates the conditioning prompt from the target to generate.
//   • Training is next-token prediction over the whole sequence, but the loss is
//     masked to the target side only: every position that would predict a prompt
//     token gets target == IgnoreIndex, so the model is scored solely on producing
//     the translation (and the closing EOS).
// ─────────────────────────────────────────────────────────────────────────────

public static class TranslationFormat
{
    /// <summary>Target id meaning "no loss here" (matches Ag.CrossEntropy's ignoreIndex).</summary>
    public const int IgnoreIndex = -1;

    /// <summary>Builds the conditioning prompt "&gt;&gt;tag&lt;&lt; source\n" as text.</summary>
    public static string PromptText(string source, string targetLangTag)
        => $">>{targetLangTag}<< {source}\n";

    /// <summary>
    /// Builds a teacher-forcing training example: returns (tokens, targets) where
    /// tokens = sequence[:-1], targets = sequence[1:], and every prompt-side target
    /// is replaced with <see cref="IgnoreIndex"/>.
    /// </summary>
    public static (int[] tokens, int[] targets) BuildExample(
        BpeTokenizer tok, string source, string target, string targetLangTag)
    {
        var promptIds = tok.Encode(PromptText(source, targetLangTag));
        var targetIds = tok.Encode(target);

        var full = new List<int>(promptIds.Count + targetIds.Count + 2) { tok.Bos };
        full.AddRange(promptIds);
        full.AddRange(targetIds);
        full.Add(tok.Eos);

        int n = full.Count;
        var tokens  = new int[n - 1];
        var targets = new int[n - 1];
        for (int i = 0; i < n - 1; i++) { tokens[i] = full[i]; targets[i] = full[i + 1]; }

        // Mask every position that predicts a prompt token: positions 0..P-1, where
        // P = promptIds.Count (targets[0..P-1] are BOS+prompt tokens). The first kept
        // position is P, which predicts the first target token from the last prompt token.
        for (int i = 0; i < promptIds.Count; i++) targets[i] = IgnoreIndex;

        return (tokens, targets);
    }

    /// <summary>
    /// Builds the inference prompt tokens "[BOS] &gt;&gt;tag&lt;&lt; source\n" to feed the
    /// model before autoregressive decoding of the target.
    /// </summary>
    public static int[] BuildPrompt(BpeTokenizer tok, string source, string targetLangTag)
    {
        var ids = new List<int> { tok.Bos };
        ids.AddRange(tok.Encode(PromptText(source, targetLangTag)));
        return [.. ids];
    }
}
