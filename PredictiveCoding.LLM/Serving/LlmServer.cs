using PredictiveCoding.Core;
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
    {
        var ids = _tok.Encode(prompt, addBos: true);
        if (ids.Count >= _model.MaxSeqLen)
            throw new ArgumentException(
                $"Prompt is {ids.Count} tokens; the window holds {_model.MaxSeqLen}. Shorten the prompt.");

        var cache = _model.CreateCache();
        DenseTensor<float> logits = null!;
        int pos = 0;
        foreach (int t in ids)
            logits = _model.Step(t, pos++, cache);

        var generated = new List<int>();
        for (int n = 0; n < maxNewTokens && pos < _model.MaxSeqLen; n++)
        {
            int next = LlmMath.Argmax(logits.Buffer);
            if (stopAtEos && next == _tok.Eos) break;
            generated.Add(next);
            logits = _model.Step(next, pos++, cache);
        }
        return _tok.Decode(generated);
    }
}
