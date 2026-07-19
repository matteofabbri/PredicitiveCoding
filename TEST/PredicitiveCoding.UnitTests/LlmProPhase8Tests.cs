using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Eval;

namespace PredicitiveCoding.UnitTests;

// Pro roadmap Phase 8: pro evaluation & observability. The multi-metric bundle,
// the Report v2 observatory, and the regression harness (the "did I make it
// worse?" button) that diffs a run against a committed baseline with a
// per-metric sense of direction.
public class LlmProPhase8Tests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"proeval_{Guid.NewGuid():N}");
    public LlmProPhase8Tests() => Directory.CreateDirectory(_dir);
    public void Dispose() { if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true); }

    private static PredictiveLlm Model()
    {
        var llm = PredictiveLlm.Create2026(dim: 16, blocks: 2, heads: 2, ffDim: 32, maxSeq: 48);
        llm.Train("the quick brown fox jumps over the lazy dog\npack my box with five dozen liquor jugs",
                  new TrainingOptions { SeqLen = 16, Stride = 8, Epochs = 8, BatchWindows = 4 });
        return llm;
    }

    private static Dictionary<string, string> Domains() => new()
    {
        ["news"] = "the quick brown fox jumps over the lazy dog",
        ["legal"] = "pack my box with five dozen liquor jugs",
    };

    // ── The bundle produces finite, well-formed metrics ──────────────────────

    [Fact]
    public void Run_ProducesAllMetrics_Finite()
    {
        var report = ProEval.Run(Model(), "the quick brown fox", Domains());

        Assert.True(double.IsFinite(report.BitsPerByte.BitsPerByte) && report.BitsPerByte.BitsPerByte > 0);
        Assert.True(report.ServeTokensPerSecond > 0);
        Assert.InRange(report.Probes.Echo, 0d, 1d);
        Assert.InRange(report.Probes.Induction, 0d, 1d);
        Assert.Equal(2, report.PerDomainPerplexity.Count);
        Assert.All(report.PerDomainPerplexity.Values, p => Assert.True(double.IsFinite(p) && p > 0));
    }

    [Fact]
    public void Flatten_NamesEveryMetric()
    {
        var flat = RegressionHarness.Flatten(ProEval.Run(Model(), "the quick brown fox", Domains()));

        Assert.Contains("bits_per_byte", flat.Keys);
        Assert.Contains("serve_tok_s", flat.Keys);
        Assert.Contains("probe_echo", flat.Keys);
        Assert.Contains("probe_induction", flat.Keys);
        Assert.Contains("ppl_news", flat.Keys);
        Assert.Contains("ppl_legal", flat.Keys);
    }

    // ── Regression harness: direction-aware diff ─────────────────────────────

    [Fact]
    public void Diff_FlagsWorseMetrics_ByDirection()
    {
        var baseline = new Dictionary<string, double>
        {
            ["bits_per_byte"]   = 2.0,   // lower is better
            ["ppl_news"]        = 10.0,  // lower is better
            ["probe_echo"]      = 0.80,  // higher is better
            ["serve_tok_s"]     = 100.0, // higher is better
        };
        var current = new Dictionary<string, double>
        {
            ["bits_per_byte"]   = 2.4,   // +20% → worse
            ["ppl_news"]        = 9.5,   // improved → fine
            ["probe_echo"]      = 0.60,  // dropped 25% → worse
            ["serve_tok_s"]     = 101.0, // ~same → fine
        };

        var regressions = RegressionHarness.Diff(baseline, current, relativeTolerance: 0.05);
        var names = regressions.Select(r => r.Metric).ToHashSet();

        Assert.Contains("bits_per_byte", names);
        Assert.Contains("probe_echo", names);
        Assert.DoesNotContain("ppl_news", names);      // improvement is never a regression
        Assert.DoesNotContain("serve_tok_s", names);   // within tolerance
        Assert.All(regressions, r => Assert.True(r.RelativeChange > 0.05));
    }

    [Fact]
    public void Diff_CleanRun_ReportsNoRegressions()
    {
        var m = new Dictionary<string, double> { ["bits_per_byte"] = 2.0, ["probe_echo"] = 0.8 };
        Assert.Empty(RegressionHarness.Diff(m, m));
        Assert.Equal("No regressions.", RegressionHarness.Summarize(RegressionHarness.Diff(m, m)));
    }

    [Fact]
    public void Baseline_RoundTripsThroughDisk()
    {
        var flat = RegressionHarness.Flatten(ProEval.Run(Model(), "the quick brown fox", Domains()));
        string path = Path.Combine(_dir, "baseline.json");
        RegressionHarness.Save(path, flat);

        var loaded = RegressionHarness.Load(path);
        Assert.Equal(flat.Count, loaded.Count);
        foreach (var (k, v) in flat) Assert.Equal(v, loaded[k], precision: 6);
        Assert.Empty(RegressionHarness.Diff(flat, loaded));   // a run against its own baseline is clean
    }

    // ── Report v2: self-contained, light theme ───────────────────────────────

    [Fact]
    public void ReportV2_IsSelfContained_AndCarriesTheMetrics()
    {
        var report = ProEval.Run(Model(), "the quick brown fox", Domains());
        string html = HtmlReport.RenderPro("Pro eval", report);

        Assert.StartsWith("<!doctype html>", html);
        Assert.Contains("bits / byte", html);
        Assert.Contains("per-domain perplexity", html, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("news", html);
        Assert.DoesNotContain("http://", html);        // no external references
        Assert.DoesNotContain("https://", html);

        string path = HtmlReport.SavePro(Path.Combine(_dir, "report.html"), "Pro eval", report);
        Assert.True(File.Exists(path));
    }
}
