using PredictiveCoding.Core;
using PredictiveCoding.LLM.Model;
using PredictiveCoding.LLM.Tokenization;

namespace PredictiveCoding.LLM.Generation;

// ─────────────────────────────────────────────────────────────────────────────
// TextGenerator  —  autoregressive text generation loop.
//
//   1. Encode the prompt and feed it through the model (prefill), building the
//      KV cache and keeping the logits of the final prompt token.
//   2. Repeatedly sample the next token, decode-step it back into the model, and
//      append — until EOS, MaxNewTokens, or the context window is full.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class TextGenerator
{
    private readonly GptModel   _model;
    private readonly ITokenizer _tokenizer;

    public TextGenerator(GptModel model, ITokenizer tokenizer)
    {
        _model     = model;
        _tokenizer = tokenizer;
    }

    /// <summary>Generates a continuation of <paramref name="prompt"/> and returns the new text.</summary>
    public string Generate(string prompt, SamplingOptions options, Action<int>? onToken = null)
    {
        var ids = GenerateTokens(_tokenizer.Encode(prompt, addBos: true), options, onToken);
        return _tokenizer.Decode(ids);
    }

    /// <summary>
    /// Token-level generation: runs prefill over <paramref name="promptTokens"/>, then
    /// samples until EOS / MaxNewTokens / MaxSeqLen.  Returns only the new tokens.
    /// </summary>
    public List<int> GenerateTokens(
        IReadOnlyList<int> promptTokens, SamplingOptions options, Action<int>? onToken = null)
    {
        var cache   = _model.CreateCache();
        var sampler = new Sampler(options.Seed);

        // Prefill: feed the prompt (or a lone BOS if empty) to build the cache and
        // obtain the logits of the final prompt token.
        IReadOnlyList<int> prompt = promptTokens.Count > 0 ? promptTokens : [_tokenizer.Bos];
        var context = new List<int>(prompt);

        int pos = 0;
        DenseTensor<float> logits = _model.Step(prompt[0], pos++, cache);
        for (int i = 1; i < prompt.Count && pos < _model.MaxSeqLen; i++)
            logits = _model.Step(prompt[i], pos++, cache);

        var generated = new List<int>();
        for (int n = 0; n < options.MaxNewTokens && pos < _model.MaxSeqLen; n++)
        {
            int next = sampler.Sample(logits, context, options);
            if (next == _tokenizer.Eos) break;

            generated.Add(next);
            context.Add(next);
            onToken?.Invoke(next);

            logits = _model.Step(next, pos++, cache);
        }
        return generated;
    }
}
