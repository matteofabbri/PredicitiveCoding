using System.Diagnostics;

namespace PredictiveCoding.LLM.Eval;

// ─────────────────────────────────────────────────────────────────────────────
// ProEval  —  the multi-metric evaluation bundle (roadmap-Pro Phase 8).
//
// One call over a trained PredictiveLlm produces the numbers a professional run
// reports: bits/byte on a reference text (the cross-tokenizer yardstick),
// per-domain perplexity, two behavioural probes (echo + induction, as
// teacher-forced next-token accuracy — the Phase-0 probes turned into scores),
// and serve throughput (tokens/s). The result flattens to a metric dictionary
// the RegressionHarness diffs against a committed baseline.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Behavioural probe scores in [0, 1] (teacher-forced next-token accuracy).</summary>
/// <param name="Echo">Predicting the next token of a constant stream (can it hold a token?).</param>
/// <param name="Induction">Predicting the next token of a 2-cycle (can it copy a period?).</param>
public sealed record ProbeScores(double Echo, double Induction);

/// <summary>Everything a pro eval run measured.</summary>
public sealed record ProEvalReport(
    BitsPerByteResult                    BitsPerByte,
    IReadOnlyDictionary<string, double>  PerDomainPerplexity,
    ProbeScores                          Probes,
    double                               ServeTokensPerSecond);

public static class ProEval
{
    /// <summary>
    /// Runs the full bundle. <paramref name="referenceText"/> feeds bits/byte and
    /// the serve-throughput timing; <paramref name="domains"/> maps a domain name
    /// to its held-out text for per-domain perplexity.
    /// </summary>
    public static ProEvalReport Run(
        PredictiveLlm llm,
        string referenceText,
        IReadOnlyDictionary<string, string> domains,
        int serveTokens = 32)
    {
        ArgumentNullException.ThrowIfNull(llm);
        ArgumentException.ThrowIfNullOrEmpty(referenceText);
        ArgumentNullException.ThrowIfNull(domains);

        var bpb = llm.EvaluateBitsPerByte(referenceText);

        var perDomain = new Dictionary<string, double>();
        foreach (var (name, text) in domains)
            perDomain[name] = llm.Evaluate(text).Perplexity;

        var probes = new ProbeScores(
            Echo:      NextTokenAccuracy(llm, Repeat("A ", 24)),
            Induction: NextTokenAccuracy(llm, Repeat("A B ", 16)));

        double tokPerSec = MeasureServeThroughput(llm, referenceText, serveTokens);

        return new ProEvalReport(bpb, perDomain, probes, tokPerSec);
    }

    // Teacher-forced fraction of positions whose argmax logit equals the actual
    // next token — a deterministic behavioural probe (no sampling).
    private static double NextTokenAccuracy(PredictiveLlm llm, string text)
    {
        var ids = llm.Tokenizer.Encode(text);
        if (ids.Count < 2) return 0d;
        int n = Math.Min(ids.Count, llm.Config.MaxSeq);
        var tokens = ids.Take(n).ToArray();

        var logits = llm.Model.Forward(tokens);            // [n, vocab]
        int vocab  = logits.Cols, correct = 0, scored = 0;
        for (int i = 0; i < n - 1; i++)
        {
            int best = 0; float bestVal = float.NegativeInfinity;
            int off = i * vocab;
            for (int v = 0; v < vocab; v++)
                if (logits.Data[off + v] > bestVal) { bestVal = logits.Data[off + v]; best = v; }
            if (best == tokens[i + 1]) correct++;
            scored++;
        }
        return scored == 0 ? 0d : correct / (double)scored;
    }

    private static double MeasureServeThroughput(PredictiveLlm llm, string referenceText, int tokens)
    {
        string prompt = referenceText.Length <= 24 ? referenceText : referenceText[..24];
        llm.Generate(prompt, maxNewTokens: 4);             // warm up
        var sw = Stopwatch.StartNew();
        llm.Generate(prompt, maxNewTokens: tokens);
        sw.Stop();
        double secs = sw.Elapsed.TotalSeconds;
        return secs > 0 ? tokens / secs : 0d;
    }

    private static string Repeat(string unit, int times)
    {
        var sb = new System.Text.StringBuilder(unit.Length * times);
        for (int i = 0; i < times; i++) sb.Append(unit);
        return sb.ToString();
    }
}
