using System.Text.Json;
using PredictiveCoding.Brain.Viz;
using Xunit;

namespace PredictiveCoding.Brain.Tests;

/// <summary>Tests for the Thousand Brains visual front-end (HtmlVisualizer) — the renderer
/// must emit a single fully self-contained HTML page with the demo data embedded as JSON:
/// no external resources, safe embedding, faithful camelCase round-trip.</summary>
public class BrainVizTests
{
    // ── helpers ─────────────────────────────────────────────────────────────

    /// <summary>A small VizData with all four sections populated.</summary>
    private static VizData BuildRichData() => new()
    {
        Title = "Thousand Brains — demo",
        Benchmark =
        [
            new VizBenchmarkRow("clean", 1.0, 3.2, 1.0, 2.1),
            new VizBenchmarkRow("occlusion", 0.97, 4.5, 1.0, 2.9),
        ],
        Episode = new VizEpisode
        {
            TargetId = 2,
            ObjectCount = 4,
            ColumnsVoting = 3,
            Objects =
            [
                new VizObject(0, [new VizPoint(0f, 0f, 1), new VizPoint(1f, 0f, 2)]),
                new VizObject(1, [new VizPoint(0f, 1f, 3), new VizPoint(1f, 1f, 4)]),
            ],
            SingleSteps = [[0, 1, 2], [0, 2], [2]],
            VotingRounds = [[0, 2], [2]],
        },
        PoseInvariance = new VizPoseInvariance
        {
            Dx = 6f,
            Dy = -4f,
            ThetaDeg = 75f,
            Rows = [new VizPoseRow(0, false, true), new VizPoseRow(1, true, true)],
        },
        ImageScan = new VizImageScan
        {
            Width = 4,
            Height = 4,
            Pixels = [.. Enumerable.Range(0, 16).Select(i => i / 15f)],
            GridCols = 2,
            GridRows = 2,
            PatchSize = 2,
            Stride = 2,
            OrientationBins = 4,
            Fixations = [new VizFixation(0, 0, 3, 0.7f, 0.4f), new VizFixation(1, 0, 8, 0.5f, 0.25f)],
        },
    };

    /// <summary>Counts case-insensitive occurrences of <paramref name="needle"/>.</summary>
    private static int CountOccurrences(string haystack, string needle)
    {
        int count = 0;
        for (int i = haystack.IndexOf(needle, StringComparison.OrdinalIgnoreCase);
             i >= 0;
             i = haystack.IndexOf(needle, i + needle.Length, StringComparison.OrdinalIgnoreCase))
            count++;
        return count;
    }

    /// <summary>Extracts the embedded DATA object literal: finds "const DATA", then the first
    /// '{' and brace-matches to its close (string- and escape-aware, so braces inside JSON
    /// strings do not fool the matcher).</summary>
    private static string ExtractDataJson(string html)
    {
        int anchor = html.IndexOf("const DATA", StringComparison.Ordinal);
        Assert.True(anchor >= 0, "\"const DATA\" not found in the rendered page.");
        int start = html.IndexOf('{', anchor);
        Assert.True(start >= 0, "no '{' found after \"const DATA\".");

        int depth = 0;
        bool inString = false;
        for (int i = start; i < html.Length; i++)
        {
            char c = html[i];
            if (inString)
            {
                if (c == '\\') i++;                       // skip the escaped character
                else if (c == '"') inString = false;
            }
            else if (c == '"') inString = true;
            else if (c == '{') depth++;
            else if (c == '}' && --depth == 0)
                return html.Substring(start, i - start + 1);
        }
        throw new InvalidOperationException("Unbalanced braces in the embedded DATA JSON.");
    }

    // ── tests ───────────────────────────────────────────────────────────────

    [Fact]
    public void Render_MinimalData_ProducesWellFormedPage()
    {
        var data = new VizData { Title = "Minimal Brain Viz Page" };
        string html = HtmlVisualizer.Render(data);

        Assert.StartsWith("<!DOCTYPE html", html, StringComparison.OrdinalIgnoreCase);
        Assert.Contains("Minimal Brain Viz Page", html);
        Assert.Contains("const DATA", html);

        int opens = CountOccurrences(html, "<script");
        int closes = CountOccurrences(html, "</script>");
        Assert.True(opens >= 1, "the page must contain at least one <script> block");
        Assert.Equal(opens, closes);
    }

    [Fact]
    public void Render_RichData_IsFullySelfContained()
    {
        string html = HtmlVisualizer.Render(BuildRichData());

        // The roadmap requirement: one file, zero external resources of any kind.
        Assert.DoesNotContain("http://", html, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("https://", html, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("<script src", html, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("<link ", html, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain("@import", html, StringComparison.OrdinalIgnoreCase);
    }

    [Fact]
    public void Render_EmbeddedJson_RoundTripsCamelCaseProperties()
    {
        var data = new VizData
        {
            Title = "Mille cervelli — sensorimotor ✓",
            Benchmark = [new VizBenchmarkRow("occlusion", 0.97, 4.5, 1.0, 2.9)],
            Episode = new VizEpisode
            {
                TargetId = 7,
                ObjectCount = 6,
                ColumnsVoting = 3,
                SingleSteps = [[1, 2, 3], [1, 2], [2]],
            },
            ImageScan = new VizImageScan
            {
                Width = 4,
                Height = 4,
                Pixels = [.. Enumerable.Range(0, 16).Select(i => i / 15f)],
                GridCols = 1,
                GridRows = 1,
                PatchSize = 4,
                Stride = 4,
                OrientationBins = 4,
                Fixations = [new VizFixation(1, 0, 8, 0.5f, 0.25f)],
            },
        };

        string html = HtmlVisualizer.Render(data);
        using var doc = JsonDocument.Parse(ExtractDataJson(html));
        var root = doc.RootElement;

        Assert.Equal("Mille cervelli — sensorimotor ✓", root.GetProperty("title").GetString());
        Assert.Equal(7, root.GetProperty("episode").GetProperty("targetId").GetInt32());
        Assert.Equal(6, root.GetProperty("episode").GetProperty("objectCount").GetInt32());

        var steps = root.GetProperty("episode").GetProperty("singleSteps");
        Assert.Equal(3, steps.GetArrayLength());
        Assert.Equal(3, steps[0].GetArrayLength());
        Assert.Equal(2, steps[0][1].GetInt32());

        var row = root.GetProperty("benchmark")[0];
        Assert.Equal("occlusion", row.GetProperty("condition").GetString());
        Assert.Equal(0.97, row.GetProperty("singleAccuracy").GetDouble(), precision: 10);

        var scan = root.GetProperty("imageScan");
        Assert.Equal(16, scan.GetProperty("pixels").GetArrayLength());
        Assert.Equal(8, scan.GetProperty("fixations")[0].GetProperty("category").GetInt32());
    }

    [Fact]
    public void Render_ScriptBreakingTitle_IsEscapedInEmbedding()
    {
        const string hostile = "</script><script>alert(1)</script>";
        string html = HtmlVisualizer.Render(new VizData { Title = hostile });

        // The default JavaScriptEncoder escapes '<', so the raw sequence can never
        // appear inside the DATA embedding — nor anywhere else unescaped.
        Assert.DoesNotContain("</script><script>", html, StringComparison.OrdinalIgnoreCase);
        Assert.DoesNotContain(hostile, html);

        // And the escaping must be lossless: the title still round-trips through JSON.
        using var doc = JsonDocument.Parse(ExtractDataJson(html));
        Assert.Equal(hostile, doc.RootElement.GetProperty("title").GetString());
    }

    [Fact]
    public void Save_WritesRenderOutputToDisk()
    {
        var data = BuildRichData();
        string path = Path.Combine(Path.GetTempPath(), $"brain-viz-{Guid.NewGuid():N}.html");
        try
        {
            HtmlVisualizer.Save(path, data);

            Assert.True(File.Exists(path));
            Assert.Equal(HtmlVisualizer.Render(data), File.ReadAllText(path));
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void VizRecords_IdenticalValues_AreEqual()
    {
        Assert.Equal(new VizPoint(1.5f, -2f, 3), new VizPoint(1.5f, -2f, 3));
        Assert.Equal(
            new VizBenchmarkRow("occlusion", 0.97, 4.5, 1.0, 2.9),
            new VizBenchmarkRow("occlusion", 0.97, 4.5, 1.0, 2.9));
        Assert.NotEqual(new VizPoint(1.5f, -2f, 3), new VizPoint(1.5f, -2f, 4));
    }
}
