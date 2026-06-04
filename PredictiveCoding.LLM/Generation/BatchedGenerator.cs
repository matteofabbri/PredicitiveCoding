using PredictiveCoding.Core;
using PredictiveCoding.LLM.Model;
using PredictiveCoding.LLM.Tokenization;

namespace PredictiveCoding.LLM.Generation;

// ─────────────────────────────────────────────────────────────────────────────
// BatchedGenerator  —  decode several independent prompts in lockstep.
//
// Each sequence keeps its own KV cache and sampler, so the batch is just N
// parallel autoregressive loops advanced one step at a time until every sequence
// hits EOS / MaxNewTokens / the context limit. This is the building block a
// serving layer would schedule continuous batches on.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class BatchedGenerator
{
    private readonly GptModel   _model;
    private readonly ITokenizer _tokenizer;

    public BatchedGenerator(GptModel model, ITokenizer tokenizer)
    {
        _model     = model;
        _tokenizer = tokenizer;
    }

    /// <summary>Generates continuations for each prompt string.</summary>
    public List<string> Generate(IReadOnlyList<string> prompts, SamplingOptions options)
    {
        var encoded = prompts.Select(p => _tokenizer.Encode(p, addBos: true)).ToList();
        var outs    = GenerateBatch(encoded, options);
        return outs.Select(t => _tokenizer.Decode(t)).ToList();
    }

    /// <summary>Token-level batched decoding; returns the new tokens per sequence.</summary>
    public List<List<int>> GenerateBatch(IReadOnlyList<IReadOnlyList<int>> prompts, SamplingOptions options)
    {
        int n = prompts.Count;
        var caches   = new KvCache[n];
        var samplers = new Sampler[n];
        var logits   = new DenseTensor<float>[n];
        var context  = new List<int>[n];
        var output   = new List<List<int>>(n);
        var pos      = new int[n];
        var done     = new bool[n];

        // Prefill each sequence.
        for (int i = 0; i < n; i++)
        {
            caches[i]   = _model.CreateCache();
            samplers[i] = new Sampler(options.Seed is { } s ? s + i : null);
            output.Add([]);

            IReadOnlyList<int> prompt = prompts[i].Count > 0 ? prompts[i] : [_tokenizer.Bos];
            context[i] = new List<int>(prompt);
            pos[i] = 0;
            logits[i] = _model.Step(prompt[0], pos[i]++, caches[i]);
            for (int t = 1; t < prompt.Count && pos[i] < _model.MaxSeqLen; t++)
                logits[i] = _model.Step(prompt[t], pos[i]++, caches[i]);
        }

        // Decode in lockstep.
        for (int step = 0; step < options.MaxNewTokens; step++)
        {
            bool anyActive = false;
            for (int i = 0; i < n; i++)
            {
                if (done[i]) continue;
                anyActive = true;

                int next = samplers[i].Sample(logits[i], context[i], options);
                if (next == _tokenizer.Eos || pos[i] >= _model.MaxSeqLen) { done[i] = true; continue; }

                output[i].Add(next);
                context[i].Add(next);
                logits[i] = _model.Step(next, pos[i]++, caches[i]);
            }
            if (!anyActive) break;
        }

        return output;
    }
}
