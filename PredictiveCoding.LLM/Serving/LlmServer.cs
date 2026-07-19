using PredictiveCoding.Core;
using PredictiveCoding.LLM.Generation;
using PredictiveCoding.LLM.Model;
using PredictiveCoding.LLM.Tokenization;

namespace PredictiveCoding.LLM.Serving;

// ─────────────────────────────────────────────────────────────────────────────
// LlmServer  —  the O(n) serving side of PredictiveLlm (ROADMAP-Pro Phase 1).
//
// Decodes incrementally through the GptModel KV cache: ONE forward per token,
// against the trainable stack's full re-forward per token. Greedy for now —
// sampling (and speculative serving) arrive with Phase 2.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class LlmServer
{
    private readonly GptModel   _model;
    private readonly ITokenizer _tok;

    internal LlmServer(GptModel model, ITokenizer tokenizer)
    {
        _model = model;
        _tok   = tokenizer;
    }

    public int MaxSeqLen => _model.MaxSeqLen;

    /// <summary>The exported inference model (KV cache, window/sink/Half features per its config).</summary>
    public GptModel Model => _model;

    /// <summary>
    /// Incremental greedy generation: the prompt is fed once through the cache,
    /// then each new token costs exactly one forward step.
    /// </summary>
    public string Generate(string prompt, int maxNewTokens = 64, bool stopAtEos = true)
        => Generate(prompt, maxNewTokens, new GenerationOptions { StopAtEos = stopAtEos });

    /// <summary>
    /// Incremental generation with the product's decoding knobs (Pro roadmap
    /// Phase 2): <see cref="GenerationOptions.Sampling"/> for stochastic
    /// decoding (null = greedy), stop sequences end the loop early. Plan
    /// settling and LearnFromPrompt are trainable-side features and are
    /// ignored here — the server is a frozen snapshot.
    /// </summary>
    public string Generate(string prompt, int maxNewTokens, GenerationOptions options)
    {
        var ids = _tok.Encode(prompt, addBos: true);
        if (ids.Count >= _model.MaxSeqLen)
            throw new ArgumentException(
                $"Prompt is {ids.Count} tokens; the window holds {_model.MaxSeqLen}. Shorten the prompt.");

        var sampler = options.Sampling is { } so ? new Sampler(so.Seed ?? 42) : null;

        var cache = _model.CreateCache();
        DenseTensor<float> logits = null!;
        int pos = 0;
        var seen = new List<int>(ids);
        foreach (int t in ids)
            logits = _model.Step(t, pos++, cache);

        var generated = new List<int>();
        string text = "";
        for (int n = 0; n < maxNewTokens && pos < _model.MaxSeqLen; n++)
        {
            int next = sampler is null
                ? LlmMath.Argmax(logits.Buffer)
                : sampler.Sample(logits, seen, options.Sampling!);
            if (options.StopAtEos && next == _tok.Eos) break;
            generated.Add(next);
            seen.Add(next);

            text = _tok.Decode(generated);
            if (Cut(ref text, options.StopSequences)) return text;

            logits = _model.Step(next, pos++, cache);
        }
        return text;
    }

    private static bool Cut(ref string text, string[]? stops)
    {
        if (stops is null) return false;
        int cut = -1;
        foreach (string s in stops)
        {
            if (string.IsNullOrEmpty(s)) continue;
            int i = text.IndexOf(s, StringComparison.Ordinal);
            if (i >= 0 && (cut < 0 || i < cut)) cut = i;
        }
        if (cut < 0) return false;
        text = text[..cut];
        return true;
    }
}
