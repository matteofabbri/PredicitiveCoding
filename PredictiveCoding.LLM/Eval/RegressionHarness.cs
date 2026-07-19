using System.Globalization;
using System.Text.Json;

namespace PredictiveCoding.LLM.Eval;

// ─────────────────────────────────────────────────────────────────────────────
// RegressionHarness  —  the "did I make it worse?" button (roadmap-Pro Phase 8).
//
// Flattens a ProEvalReport to a flat metric dictionary, persists it as JSON
// (the committed baseline), and diffs a fresh run against that baseline with a
// per-metric sense of direction: perplexity and bits/byte are LOWER-is-better,
// probe accuracies and serve throughput are HIGHER-is-better. A relative move
// past the tolerance in the wrong direction is a regression.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>One metric that moved the wrong way beyond tolerance.</summary>
/// <param name="Metric">Metric name.</param>
/// <param name="Baseline">Committed value.</param>
/// <param name="Current">This run's value.</param>
/// <param name="RelativeChange">Signed relative change toward "worse" (positive = worse).</param>
public sealed record MetricRegression(string Metric, double Baseline, double Current, double RelativeChange);

public static class RegressionHarness
{
    /// <summary>Flattens a report to <c>name → value</c>. Lower-is-better metrics keep their raw value; direction is applied at diff time.</summary>
    public static Dictionary<string, double> Flatten(ProEvalReport report)
    {
        var m = new Dictionary<string, double>
        {
            ["bits_per_byte"]   = report.BitsPerByte.BitsPerByte,
            ["probe_echo"]      = report.Probes.Echo,
            ["probe_induction"] = report.Probes.Induction,
            ["serve_tok_s"]     = report.ServeTokensPerSecond,
        };
        foreach (var (domain, ppl) in report.PerDomainPerplexity)
            m[$"ppl_{domain}"] = ppl;
        return m;
    }

    public static void Save(string path, IReadOnlyDictionary<string, double> metrics)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        File.WriteAllText(path, JsonSerializer.Serialize(metrics, new JsonSerializerOptions { WriteIndented = true }));
    }

    public static Dictionary<string, double> Load(string path)
        => JsonSerializer.Deserialize<Dictionary<string, double>>(File.ReadAllText(path))
           ?? throw new InvalidDataException($"Unreadable baseline metrics in '{path}'.");

    /// <summary>True when the metric improves as it goes UP (probes, throughput); false = lower is better (ppl, bits/byte).</summary>
    public static bool HigherIsBetter(string metric)
        => metric.StartsWith("probe_", StringComparison.Ordinal) || metric == "serve_tok_s";

    /// <summary>
    /// Metrics in <paramref name="current"/> that moved the wrong way past
    /// <paramref name="relativeTolerance"/> versus <paramref name="baseline"/>.
    /// Metrics absent from either side are skipped. Empty ⇒ no regressions.
    /// </summary>
    public static IReadOnlyList<MetricRegression> Diff(
        IReadOnlyDictionary<string, double> baseline,
        IReadOnlyDictionary<string, double> current,
        double relativeTolerance = 0.05)
    {
        var regressions = new List<MetricRegression>();
        foreach (var (metric, baseVal) in baseline)
        {
            if (!current.TryGetValue(metric, out double curVal)) continue;
            double denom = Math.Max(Math.Abs(baseVal), 1e-9);
            // Signed change toward "worse": +worse. Higher-is-better flips the sign.
            double worse = (HigherIsBetter(metric) ? (baseVal - curVal) : (curVal - baseVal)) / denom;
            if (worse > relativeTolerance)
                regressions.Add(new MetricRegression(metric, baseVal, curVal, worse));
        }
        regressions.Sort((a, b) => b.RelativeChange.CompareTo(a.RelativeChange));
        return regressions;
    }

    /// <summary>A one-line human summary of a diff (for the "one command" console output).</summary>
    public static string Summarize(IReadOnlyList<MetricRegression> regressions)
    {
        if (regressions.Count == 0) return "No regressions.";
        var ci = CultureInfo.InvariantCulture;
        return $"{regressions.Count} regression(s): " + string.Join(", ",
            regressions.Select(r => $"{r.Metric} {r.Baseline.ToString("G4", ci)}→{r.Current.ToString("G4", ci)} ({r.RelativeChange:P0} worse)"));
    }
}
