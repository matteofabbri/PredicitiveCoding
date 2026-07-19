using PredictiveCoding.LLM.Autograd;

namespace PredictiveCoding.LLM.Eval;

// ─────────────────────────────────────────────────────────────────────────────
// PerplexityEvaluator  —  the yardstick (roadmap Phase 0).
//
// Perplexity = exp(mean negative log-likelihood per predicted token). It is the
// prediction-error number of a language model: 1 would be a perfect predictor,
// vocab-size is "uniform guessing". Every architecture phase of the roadmap
// quotes its before/after from here.
//
// Deterministic by construction: evaluation runs the forward with
// training = false (no dropout), so the same model + windows always produce the
// same number. No autograd graph is consulted — only the logits.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Result of a perplexity evaluation.</summary>
/// <param name="Perplexity">exp(NllPerToken) — lower is better, vocab size ≈ uniform guessing.</param>
/// <param name="NllPerToken">Mean negative log-likelihood per scored token (nats).</param>
/// <param name="Tokens">Number of tokens actually scored (ignoreIndex targets are skipped).</param>
/// <param name="Windows">Number of (input, target) windows consumed.</param>
public readonly record struct PerplexityResult(double Perplexity, double NllPerToken, long Tokens, int Windows);

public static class PerplexityEvaluator
{
    /// <summary>
    /// Token-weighted perplexity of <paramref name="model"/> over
    /// <paramref name="windows"/>. Targets equal to <paramref name="ignoreIndex"/>
    /// are skipped; <paramref name="maxWindows"/> caps the work on streaming
    /// sources (null = consume everything).
    /// </summary>
    public static PerplexityResult Evaluate(
        DecoderLM                                    model,
        IEnumerable<(int[] input, int[] target)>     windows,
        int                                          ignoreIndex = -1,
        int?                                         maxWindows  = null)
    {
        double nll    = 0d;
        long   tokens = 0;
        int    used   = 0;

        foreach (var (input, target) in windows)
        {
            if (maxWindows is { } cap && used >= cap) break;

            var logits = model.Forward(input, training: false);
            int vocab  = logits.Cols;

            for (int r = 0; r < input.Length; r++)
            {
                int t = target[r];
                if (t == ignoreIndex) continue;
                nll += RowNll(logits.Data, r * vocab, vocab, t);
                tokens++;
            }
            used++;
        }

        if (tokens == 0)
            throw new InvalidOperationException(
                "No scorable tokens: the window stream was empty or every target is ignoreIndex.");

        double perToken = nll / tokens;
        return new PerplexityResult(Math.Exp(perToken), perToken, tokens, used);
    }

    /// <summary>
    /// Deterministic held-out slice of a window stream: yields every
    /// <paramref name="every"/>-th window (indices 0, every, 2·every, …).
    /// The complement is <see cref="TrainingSplit"/>; together they partition
    /// the stream without buffering it.
    /// </summary>
    public static IEnumerable<(int[] input, int[] target)> HeldOut(
        IEnumerable<(int[] input, int[] target)> windows, int every = 10)
    {
        if (every < 2) throw new ArgumentOutOfRangeException(nameof(every), every, "every must be >= 2.");
        int i = 0;
        foreach (var w in windows)
            if (i++ % every == 0) yield return w;
    }

    /// <summary>The training-side complement of <see cref="HeldOut"/> (same <paramref name="every"/>).</summary>
    public static IEnumerable<(int[] input, int[] target)> TrainingSplit(
        IEnumerable<(int[] input, int[] target)> windows, int every = 10)
    {
        if (every < 2) throw new ArgumentOutOfRangeException(nameof(every), every, "every must be >= 2.");
        int i = 0;
        foreach (var w in windows)
            if (i++ % every != 0) yield return w;
    }

    /// <summary>
    /// Content-hashed held-out slice (roadmap-Pro Phase 4): a window is held out
    /// iff a stable hash of its input tokens is ≡ 0 (mod <paramref name="every"/>).
    /// Unlike <see cref="HeldOut"/> this is independent of position, so the SAME
    /// windows land in eval across runs even if the corpus is reordered, re-mixed
    /// or grown — perplexity numbers stay comparable between sessions. Pair with
    /// <see cref="TrainingSplitHashed"/> for the exact complement.
    /// </summary>
    public static IEnumerable<(int[] input, int[] target)> HeldOutHashed(
        IEnumerable<(int[] input, int[] target)> windows, int every = 10)
    {
        if (every < 2) throw new ArgumentOutOfRangeException(nameof(every), every, "every must be >= 2.");
        foreach (var w in windows)
            if (WindowHash(w.input) % (uint)every == 0) yield return w;
    }

    /// <summary>The training-side complement of <see cref="HeldOutHashed"/> (same <paramref name="every"/>).</summary>
    public static IEnumerable<(int[] input, int[] target)> TrainingSplitHashed(
        IEnumerable<(int[] input, int[] target)> windows, int every = 10)
    {
        if (every < 2) throw new ArgumentOutOfRangeException(nameof(every), every, "every must be >= 2.");
        foreach (var w in windows)
            if (WindowHash(w.input) % (uint)every != 0) yield return w;
    }

    // FNV-1a 32-bit over the window's token ids — stable across sessions/platforms.
    private static uint WindowHash(int[] tokens)
    {
        const uint offset = 2166136261u, prime = 16777619u;
        uint h = offset;
        foreach (int t in tokens)
        {
            uint v = (uint)t;
            for (int b = 0; b < 4; b++) { h ^= (v >> (b * 8)) & 0xFFu; h *= prime; }
        }
        return h;
    }

    // Stable −log softmax(logits)[target]: log-sum-exp − logit_target.
    private static double RowNll(float[] data, int off, int vocab, int target)
    {
        float max = float.NegativeInfinity;
        for (int j = 0; j < vocab; j++) max = MathF.Max(max, data[off + j]);
        double sum = 0d;
        for (int j = 0; j < vocab; j++) sum += Math.Exp(data[off + j] - max);
        return Math.Log(sum) + max - data[off + target];
    }
}
