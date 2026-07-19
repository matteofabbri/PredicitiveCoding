using System.Globalization;
using System.Text;

namespace PredictiveCoding.LLM.Eval;

// ─────────────────────────────────────────────────────────────────────────────
// HtmlReport  —  the Phase 0 observatory as one self-contained HTML file.
//
// Same philosophy as the Brain project's HtmlVisualizer: a single file, inline
// CSS, no server, no external references, light theme. Open it, read the
// numbers, close it.
// ─────────────────────────────────────────────────────────────────────────────

public static class HtmlReport
{
    /// <summary>Renders perplexity + activation-observatory readings as a standalone HTML page.</summary>
    public static string Render(string title, PerplexityResult? perplexity, ActivationProbe? probe)
    {
        var ci = CultureInfo.InvariantCulture;
        var sb = new StringBuilder();

        sb.Append("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
        sb.Append("<title>").Append(Escape(title)).Append("</title>\n");
        sb.Append("<style>\n");
        sb.Append("body{font-family:'Segoe UI',system-ui,sans-serif;background:#ffffff;color:#1f2328;margin:2rem auto;max-width:900px;padding:0 1rem;}\n");
        sb.Append("h1{font-size:1.5rem;border-bottom:1px solid #d1d9e0;padding-bottom:.4rem;}\n");
        sb.Append("h2{font-size:1.15rem;margin-top:1.6rem;}\n");
        sb.Append(".cards{display:flex;gap:1rem;flex-wrap:wrap;}\n");
        sb.Append(".card{border:1px solid #d1d9e0;border-radius:8px;padding:.8rem 1.2rem;background:#f6f8fa;}\n");
        sb.Append(".card b{display:block;font-size:1.3rem;color:#0969da;}\n");
        sb.Append("table{border-collapse:collapse;margin-top:.5rem;}\n");
        sb.Append("td,th{border:1px solid #d1d9e0;padding:.3rem .7rem;text-align:right;font-variant-numeric:tabular-nums;}\n");
        sb.Append("th{background:#f6f8fa;text-align:center;}\n");
        sb.Append(".bar{display:inline-block;height:.6rem;background:#0969da;border-radius:2px;vertical-align:middle;}\n");
        sb.Append("</style>\n</head>\n<body>\n");
        sb.Append("<h1>").Append(Escape(title)).Append("</h1>\n");

        if (perplexity is { } p)
        {
            sb.Append("<h2>Perplexity</h2>\n<div class=\"cards\">\n");
            Card(sb, "Perplexity", p.Perplexity.ToString("F2", ci));
            Card(sb, "NLL / token (nats)", p.NllPerToken.ToString("F3", ci));
            Card(sb, "Tokens scored", p.Tokens.ToString(ci));
            Card(sb, "Windows", p.Windows.ToString(ci));
            sb.Append("</div>\n");
        }

        if (probe is not null)
        {
            if (probe.LoopsUsed > 0)
                sb.Append("<p>Loop passes used by the last forward: <b>")
                  .Append(probe.LoopsUsed).Append("</b></p>\n");

            if (probe.Blocks.Count > 0)
            {
                sb.Append("<h2>Block activations (spike watch)</h2>\n");
                sb.Append("<table>\n<tr><th>Block</th><th>mean |x|</th><th>max |x|</th><th>spike ratio</th></tr>\n");
                foreach (var b in probe.Blocks)
                    sb.Append("<tr><td>").Append(b.Block)
                      .Append("</td><td>").Append(b.MeanAbs.ToString("F4", ci))
                      .Append("</td><td>").Append(b.MaxAbs.ToString("F4", ci))
                      .Append("</td><td>").Append(b.SpikeRatio.ToString("F1", ci))
                      .Append("</td></tr>\n");
                sb.Append("</table>\n");
            }

            if (probe.Sinks.Count > 0)
            {
                sb.Append("<h2>Attention-sink share (mass on position 0)</h2>\n");
                sb.Append("<table>\n<tr><th>Block</th><th>Head</th><th>share</th><th></th></tr>\n");
                foreach (var s in probe.Sinks)
                    sb.Append("<tr><td>").Append(s.Block)
                      .Append("</td><td>").Append(s.Head)
                      .Append("</td><td>").Append(s.SinkShare.ToString("F3", ci))
                      .Append("</td><td style=\"text-align:left;min-width:120px\"><span class=\"bar\" style=\"width:")
                      .Append((s.SinkShare * 100f).ToString("F0", ci))
                      .Append("px\"></span></td></tr>\n");
                sb.Append("</table>\n");
            }
        }

        sb.Append("</body>\n</html>\n");
        return sb.ToString();
    }

    /// <summary>Renders and writes the report; returns the path.</summary>
    public static string Save(string path, string title, PerplexityResult? perplexity, ActivationProbe? probe)
    {
        File.WriteAllText(path, Render(title, perplexity, probe));
        return path;
    }

    /// <summary>
    /// Report v2 (roadmap-Pro Phase 8): the multi-metric observatory — bits/byte,
    /// per-domain perplexity, behavioural probes and serve throughput — as one
    /// self-contained light-theme page. An optional <paramref name="probe"/>
    /// appends the activation/sink tables from <see cref="Render"/>.
    /// </summary>
    public static string RenderPro(string title, ProEvalReport report, ActivationProbe? probe = null)
    {
        var ci = CultureInfo.InvariantCulture;
        var sb = new StringBuilder();

        sb.Append("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n");
        sb.Append("<title>").Append(Escape(title)).Append("</title>\n");
        sb.Append("<style>\n");
        sb.Append("body{font-family:'Segoe UI',system-ui,sans-serif;background:#ffffff;color:#1f2328;margin:2rem auto;max-width:900px;padding:0 1rem;}\n");
        sb.Append("h1{font-size:1.5rem;border-bottom:1px solid #d1d9e0;padding-bottom:.4rem;}\n");
        sb.Append("h2{font-size:1.15rem;margin-top:1.6rem;}\n");
        sb.Append(".cards{display:flex;gap:1rem;flex-wrap:wrap;}\n");
        sb.Append(".card{border:1px solid #d1d9e0;border-radius:8px;padding:.8rem 1.2rem;background:#f6f8fa;}\n");
        sb.Append(".card b{display:block;font-size:1.3rem;color:#0969da;}\n");
        sb.Append("table{border-collapse:collapse;margin-top:.5rem;}\n");
        sb.Append("td,th{border:1px solid #d1d9e0;padding:.3rem .7rem;text-align:right;font-variant-numeric:tabular-nums;}\n");
        sb.Append("th{background:#f6f8fa;text-align:center;}\n");
        sb.Append(".bar{display:inline-block;height:.6rem;background:#0969da;border-radius:2px;vertical-align:middle;}\n");
        sb.Append("</style>\n</head>\n<body>\n");
        sb.Append("<h1>").Append(Escape(title)).Append("</h1>\n");

        sb.Append("<h2>Headline metrics</h2>\n<div class=\"cards\">\n");
        Card(sb, "bits / byte", report.BitsPerByte.BitsPerByte.ToString("F3", ci));
        Card(sb, "serve tokens / s", report.ServeTokensPerSecond.ToString("F1", ci));
        Card(sb, "probe: echo", report.Probes.Echo.ToString("P0", ci));
        Card(sb, "probe: induction", report.Probes.Induction.ToString("P0", ci));
        sb.Append("</div>\n");

        if (report.PerDomainPerplexity.Count > 0)
        {
            sb.Append("<h2>Per-domain perplexity</h2>\n<table>\n<tr><th>Domain</th><th>perplexity</th></tr>\n");
            foreach (var (domain, ppl) in report.PerDomainPerplexity)
                sb.Append("<tr><td style=\"text-align:left\">").Append(Escape(domain))
                  .Append("</td><td>").Append(ppl.ToString("F2", ci)).Append("</td></tr>\n");
            sb.Append("</table>\n");
        }

        if (probe is not null && probe.Blocks.Count > 0)
        {
            sb.Append("<h2>Block activations (spike watch)</h2>\n");
            sb.Append("<table>\n<tr><th>Block</th><th>mean |x|</th><th>max |x|</th><th>spike ratio</th></tr>\n");
            foreach (var b in probe.Blocks)
                sb.Append("<tr><td>").Append(b.Block)
                  .Append("</td><td>").Append(b.MeanAbs.ToString("F4", ci))
                  .Append("</td><td>").Append(b.MaxAbs.ToString("F4", ci))
                  .Append("</td><td>").Append(b.SpikeRatio.ToString("F1", ci))
                  .Append("</td></tr>\n");
            sb.Append("</table>\n");
        }

        sb.Append("</body>\n</html>\n");
        return sb.ToString();
    }

    /// <summary>Renders and writes the Report v2; returns the path.</summary>
    public static string SavePro(string path, string title, ProEvalReport report, ActivationProbe? probe = null)
    {
        File.WriteAllText(path, RenderPro(title, report, probe));
        return path;
    }

    private static void Card(StringBuilder sb, string label, string value)
        => sb.Append("<div class=\"card\"><b>").Append(value).Append("</b>").Append(Escape(label)).Append("</div>\n");

    private static string Escape(string s)
        => s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;");
}
