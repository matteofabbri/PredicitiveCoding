using PredictiveCoding.Core;
using PredictiveCoding.LLM.Model;
using PredictiveCoding.LLM.Tokenization;

namespace PredictiveCoding.LLM.Generation;

// ─────────────────────────────────────────────────────────────────────────────
// ContinuousBatcher  —  in-process serving engine with continuous batching.
//
// Unlike static batching (which waits for the whole batch to finish), this
// scheduler keeps up to MaxConcurrency sequences live: each scheduler tick
// advances every active sequence by one token, evicts the ones that just
// finished, and immediately admits queued requests into the freed slots. This is
// the core loop a token-streaming server schedules on (the network/API layer is
// out of scope).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class ContinuousBatcher
{
    private readonly GptModel   _model;
    private readonly ITokenizer _tokenizer;
    private readonly int        _maxConcurrency;

    public ContinuousBatcher(GptModel model, ITokenizer tokenizer, int maxConcurrency)
    {
        if (maxConcurrency < 1) throw new ArgumentOutOfRangeException(nameof(maxConcurrency));
        _model          = model;
        _tokenizer      = tokenizer;
        _maxConcurrency = maxConcurrency;
    }

    private sealed class Seq
    {
        public required int                Id;
        public required KvCache            Cache;
        public required SamplingOptions    Options;
        public required Sampler            Sampler;
        public required List<int>          Context;
        public required DenseTensor<float> Logits;
        public required int                Position;
        public readonly List<int>          Generated = [];
        public bool                        Done;
    }

    /// <summary>Runs all token-level requests to completion; returns new tokens per request (in order).</summary>
    public List<List<int>> Run(IReadOnlyList<(IReadOnlyList<int> prompt, SamplingOptions options)> requests)
    {
        var results = new List<int>[requests.Count];
        var pending = new Queue<int>(Enumerable.Range(0, requests.Count));
        var active  = new List<Seq>();

        while (pending.Count > 0 || active.Count > 0)
        {
            // Admit queued requests into free slots (continuous batching).
            while (active.Count < _maxConcurrency && pending.Count > 0)
                active.Add(Prefill(pending.Dequeue(), requests));

            // Advance every active sequence by one token.
            foreach (var seq in active)
            {
                int next = seq.Sampler.Sample(seq.Logits, seq.Context, seq.Options);
                if (next == _tokenizer.Eos)
                {
                    seq.Done = true;
                    continue;
                }
                seq.Generated.Add(next);
                seq.Context.Add(next);
                seq.Logits = _model.Step(next, seq.Position++, seq.Cache);
                if (seq.Generated.Count >= seq.Options.MaxNewTokens || seq.Position >= _model.MaxSeqLen)
                    seq.Done = true;
            }

            // Evict finished sequences and record their output (freeing slots).
            for (int i = active.Count - 1; i >= 0; i--)
                if (active[i].Done)
                {
                    results[active[i].Id] = active[i].Generated;
                    active.RemoveAt(i);
                }
        }

        return results.Select(r => r ?? []).ToList();
    }

    /// <summary>String convenience overload.</summary>
    public List<string> Run(IReadOnlyList<string> prompts, SamplingOptions options)
    {
        var reqs = prompts
            .Select(p => ((IReadOnlyList<int>)_tokenizer.Encode(p, addBos: true), options))
            .ToList();
        return Run(reqs).Select(t => _tokenizer.Decode(t)).ToList();
    }

    private Seq Prefill(int id, IReadOnlyList<(IReadOnlyList<int> prompt, SamplingOptions options)> requests)
    {
        var (promptTokens, options) = requests[id];
        IReadOnlyList<int> prompt = promptTokens.Count > 0 ? promptTokens : [_tokenizer.Bos];

        var cache = _model.CreateCache();
        int pos   = 0;
        DenseTensor<float> logits = _model.Step(prompt[0], pos++, cache);
        for (int t = 1; t < prompt.Count && pos < _model.MaxSeqLen; t++)
            logits = _model.Step(prompt[t], pos++, cache);

        return new Seq
        {
            Id = id, Cache = cache, Options = options,
            Sampler = new Sampler(options.Seed is { } s ? s + id : null),
            Context = new List<int>(prompt), Logits = logits, Position = pos,
        };
    }
}
