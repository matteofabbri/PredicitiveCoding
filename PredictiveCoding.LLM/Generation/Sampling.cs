using PredictiveCoding.Core;
using PredictiveCoding.LLM.Model;

namespace PredictiveCoding.LLM.Generation;

// ─────────────────────────────────────────────────────────────────────────────
// Sampling  —  turns next-token logits into a chosen token id.
//
// Supports greedy decoding and stochastic sampling with temperature, top-k,
// top-p (nucleus) truncation and a repetition penalty.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record SamplingOptions
{
    /// <summary>Softmax temperature (&gt;0). Lower = sharper / more deterministic.</summary>
    public float Temperature { get; init; } = 1.0f;

    /// <summary>Keep only the K highest-probability tokens (0 = disabled).</summary>
    public int TopK { get; init; } = 0;

    /// <summary>Nucleus sampling: keep the smallest set whose cumulative prob ≥ TopP (1 = disabled).</summary>
    public float TopP { get; init; } = 1.0f;

    /// <summary>Penalty applied to already-generated tokens (1 = none, &gt;1 discourages repeats).</summary>
    public float RepetitionPenalty { get; init; } = 1.0f;

    /// <summary>Subtracted from a token's logit once it has appeared (OpenAI presence penalty).</summary>
    public float PresencePenalty { get; init; } = 0.0f;

    /// <summary>Subtracted from a token's logit per prior occurrence (OpenAI frequency penalty).</summary>
    public float FrequencyPenalty { get; init; } = 0.0f;

    /// <summary>Greedy (argmax) decoding, ignoring the stochastic settings.</summary>
    public bool Greedy { get; init; } = false;

    /// <summary>Maximum number of tokens to generate.</summary>
    public int MaxNewTokens { get; init; } = 128;

    /// <summary>Optional RNG seed for reproducible sampling.</summary>
    public int? Seed { get; init; }
}

public sealed class Sampler
{
    private readonly Random _rng;

    public Sampler(int? seed = null) => _rng = seed is { } s ? new Random(s) : new Random();

    /// <summary>Selects the next token id from <paramref name="logits"/>.</summary>
    public int Sample(DenseTensor<float> logits, IReadOnlyCollection<int> generated, SamplingOptions o)
    {
        var l = (float[])logits.Buffer.Clone();

        // Repetition / presence / frequency penalties on tokens already emitted.
        bool hasPenalty = o.RepetitionPenalty != 1f || o.PresencePenalty != 0f || o.FrequencyPenalty != 0f;
        if (hasPenalty && generated.Count > 0)
        {
            var counts = new Dictionary<int, int>();
            foreach (int t in generated)
                if (t >= 0 && t < l.Length) counts[t] = counts.GetValueOrDefault(t) + 1;

            foreach (var (t, c) in counts)
            {
                if (o.RepetitionPenalty != 1f)
                    l[t] = l[t] > 0f ? l[t] / o.RepetitionPenalty : l[t] * o.RepetitionPenalty;
                l[t] -= o.PresencePenalty + o.FrequencyPenalty * c;
            }
        }

        if (o.Greedy) return Argmax(l);

        float temp = MathF.Max(o.Temperature, 1e-5f);
        for (int i = 0; i < l.Length; i++) l[i] /= temp;

        // logits → probabilities
        Softmax(l);

        if (o.TopK > 0)   ApplyTopK(l, o.TopK);
        if (o.TopP < 1f)  ApplyTopP(l, o.TopP);

        Renormalize(l);
        return SampleMultinomial(l);
    }

    // ── helpers ────────────────────────────────────────────────────────────────

    private static int Argmax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static void Softmax(float[] v)
    {
        float max = float.NegativeInfinity;
        for (int i = 0; i < v.Length; i++) if (v[i] > max) max = v[i];
        float sum = 0f;
        for (int i = 0; i < v.Length; i++) { v[i] = MathF.Exp(v[i] - max); sum += v[i]; }
        float inv = sum > 0f ? 1f / sum : 0f;
        for (int i = 0; i < v.Length; i++) v[i] *= inv;
    }

    private static void ApplyTopK(float[] probs, int k)
    {
        if (k >= probs.Length) return;
        // kth-largest probability as threshold
        var sorted = (float[])probs.Clone();
        Array.Sort(sorted);                 // ascending
        float threshold = sorted[probs.Length - k];
        for (int i = 0; i < probs.Length; i++) if (probs[i] < threshold) probs[i] = 0f;
    }

    private static void ApplyTopP(float[] probs, float p)
    {
        int[] idx = Enumerable.Range(0, probs.Length).ToArray();
        Array.Sort(idx, (a, b) => probs[b].CompareTo(probs[a]));   // descending by prob
        float cum = 0f;
        bool kept = false;
        foreach (int i in idx)
        {
            if (kept && cum >= p) { probs[i] = 0f; continue; }
            cum += probs[i];
            kept = true;
        }
    }

    private static void Renormalize(float[] v)
    {
        float sum = 0f;
        for (int i = 0; i < v.Length; i++) sum += v[i];
        if (sum <= 0f) { Array.Fill(v, 1f / v.Length); return; }
        float inv = 1f / sum;
        for (int i = 0; i < v.Length; i++) v[i] *= inv;
    }

    private int SampleMultinomial(float[] probs)
    {
        double r   = _rng.NextDouble();
        double acc = 0d;
        for (int i = 0; i < probs.Length; i++)
        {
            acc += probs[i];
            if (r <= acc) return i;
        }
        return probs.Length - 1;
    }
}
