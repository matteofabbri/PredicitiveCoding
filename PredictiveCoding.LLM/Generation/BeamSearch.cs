using PredictiveCoding.Core;
using PredictiveCoding.LLM.Model;
using PredictiveCoding.LLM.Tokenization;

namespace PredictiveCoding.LLM.Generation;

// ─────────────────────────────────────────────────────────────────────────────
// BeamSearch  —  deterministic beam-search decoding.
//
// Keeps the <c>BeamWidth</c> highest-scoring hypotheses. Each step expands every
// live beam by its top-<c>BeamWidth</c> next tokens (forking its KV cache), then
// keeps the globally best beams by length-normalised log-probability. Finishes
// when all surviving beams have emitted EOS or the limits are reached.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record BeamSearchOptions
{
    public int   BeamWidth     { get; init; } = 4;
    public int   MaxNewTokens  { get; init; } = 128;
    /// <summary>Length-normalisation exponent α: score ÷ len^α (0 = no normalisation).</summary>
    public float LengthPenalty { get; init; } = 0.6f;
}

public sealed class BeamSearch
{
    private readonly GptModel   _model;
    private readonly ITokenizer _tokenizer;

    public BeamSearch(GptModel model, ITokenizer tokenizer)
    {
        _model     = model;
        _tokenizer = tokenizer;
    }

    private sealed class Beam
    {
        public required List<int>            Tokens;
        public required double               LogProb;
        public required KvCache              Cache;
        public required int                  Position;
        public DenseTensor<float>?           Logits;     // null once finished
        public bool                          Finished;
    }

    public string Search(string prompt, BeamSearchOptions options)
        => _tokenizer.Decode(SearchTokens(_tokenizer.Encode(prompt, addBos: true), options));

    public List<int> SearchTokens(IReadOnlyList<int> promptTokens, BeamSearchOptions options)
    {
        int width = Math.Max(1, options.BeamWidth);

        // Prefill the prompt into one cache.
        var cache = _model.CreateCache();
        IReadOnlyList<int> prompt = promptTokens.Count > 0 ? promptTokens : [_tokenizer.Bos];
        int pos = 0;
        DenseTensor<float> logits = _model.Step(prompt[0], pos++, cache);
        for (int i = 1; i < prompt.Count && pos < _model.MaxSeqLen; i++)
            logits = _model.Step(prompt[i], pos++, cache);

        var beams = new List<Beam>
        {
            new() { Tokens = [], LogProb = 0d, Cache = cache, Position = pos, Logits = logits, Finished = false },
        };

        for (int step = 0; step < options.MaxNewTokens; step++)
        {
            var candidates = new List<Beam>();

            foreach (var beam in beams)
            {
                if (beam.Finished || beam.Position >= _model.MaxSeqLen) { candidates.Add(beam); continue; }

                var logp = LogSoftmax(beam.Logits!.Buffer);
                foreach (int tok in TopK(logp, width))
                {
                    double score = beam.LogProb + logp[tok];
                    if (tok == _tokenizer.Eos)
                    {
                        candidates.Add(new Beam
                        {
                            Tokens = beam.Tokens, LogProb = score, Cache = beam.Cache,
                            Position = beam.Position, Logits = null, Finished = true,
                        });
                    }
                    else
                    {
                        var forked = beam.Cache.Clone();
                        var next   = _model.Step(tok, beam.Position, forked);
                        candidates.Add(new Beam
                        {
                            Tokens = [.. beam.Tokens, tok], LogProb = score, Cache = forked,
                            Position = beam.Position + 1, Logits = next, Finished = false,
                        });
                    }
                }
            }

            beams = candidates
                .OrderByDescending(b => b.LogProb / LengthNorm(b.Tokens.Count, options.LengthPenalty))
                .Take(width)
                .ToList();

            if (beams.All(b => b.Finished)) break;
        }

        return beams
            .OrderByDescending(b => b.LogProb / LengthNorm(b.Tokens.Count, options.LengthPenalty))
            .First().Tokens;
    }

    // ── helpers ────────────────────────────────────────────────────────────────

    private static double LengthNorm(int len, float alpha)
        => Math.Pow(Math.Max(1, len), alpha);

    private static float[] LogSoftmax(float[] logits)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++) if (logits[i] > max) max = logits[i];
        float sum = 0f;
        for (int i = 0; i < logits.Length; i++) sum += MathF.Exp(logits[i] - max);
        float logZ = max + MathF.Log(sum);
        var outp = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++) outp[i] = logits[i] - logZ;
        return outp;
    }

    private static IEnumerable<int> TopK(float[] v, int k)
    {
        return Enumerable.Range(0, v.Length)
                         .OrderByDescending(i => v[i])
                         .Take(Math.Min(k, v.Length));
    }
}
