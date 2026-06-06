using System.Text.Json;
using System.Text.RegularExpressions;
using PredictiveCoding.Data.ArXiv;
using PredictiveCoding.Data.ConceptNet;
using PredictiveCoding.Data.Crypto;
using PredictiveCoding.Data.Finance;
using PredictiveCoding.Data.GitHub;
using PredictiveCoding.Data.HackerNews;
using PredictiveCoding.Data.HuggingFace;
using PredictiveCoding.Data.Mastodon;
using PredictiveCoding.Data.OpenStreetMap;
using PredictiveCoding.Data.PubMed;
using PredictiveCoding.Data.Reddit;
using PredictiveCoding.Data.Rss;
using PredictiveCoding.Data.Scholarly;
using PredictiveCoding.Data.Sparql;
using PredictiveCoding.Data.StackExchange;
using PredictiveCoding.Data.Weather;
using PredictiveCoding.Data.Web;
using PredictiveCoding.Data.Wikidata;
using PredictiveCoding.Data.Wikipedia;
using PredictiveCoding.Data.WorldBank;
using PredictiveCoding.Data.YouTube;

namespace PredictiveCoding.OmniMind.Engine;

/// <summary>Lifecycle state of a single provider probe.</summary>
public enum ProbeState { Pending, Running, Ok, Empty, Skipped, Failed }

/// <summary>The outcome of probing one external source: its state, item count, the
/// text it contributed to the fused corpus, and a short human-readable detail.</summary>
public sealed record ProbeResult(ProbeState State, int ItemCount, IReadOnlyList<string> Text, string Detail)
{
    public static readonly ProbeResult Empty = new(ProbeState.Empty, 0, Array.Empty<string>(), "no results");
    public static ProbeResult Ok(IReadOnlyList<string> text, string detail) => new(ProbeState.Ok, text.Count, text, detail);
    public static ProbeResult Skip(string why) => new(ProbeState.Skipped, 0, Array.Empty<string>(), why);
}

/// <summary>One named external source the OmniMind can read, as a topic → result probe.</summary>
public sealed class KnowledgeProbe
{
    public required string Name { get; init; }
    public required string Category { get; init; }
    public required Func<string, CancellationToken, Task<ProbeResult>> RunAsync { get; init; }
}

/// <summary>Runs a single probe with a per-source timeout, turning any failure into a result.</summary>
public static class KnowledgeFanOut
{
    public static async Task<ProbeResult> ProbeAsync(
        KnowledgeProbe probe, string topic, TimeSpan timeout, CancellationToken ct)
    {
        using var cts = CancellationTokenSource.CreateLinkedTokenSource(ct);
        cts.CancelAfter(timeout);
        try
        {
            return await probe.RunAsync(topic, cts.Token).ConfigureAwait(false);
        }
        catch (OperationCanceledException) when (!ct.IsCancellationRequested)
        {
            return new ProbeResult(ProbeState.Failed, 0, Array.Empty<string>(), "timeout");
        }
        catch (Exception ex)
        {
            return new ProbeResult(ProbeState.Failed, 0, Array.Empty<string>(), Short(ex.Message));
        }
    }

    private static string Short(string s)
    {
        s = s.Replace('\n', ' ').Replace('\r', ' ').Trim();
        return s.Length <= 60 ? s : s[..60];
    }
}

/// <summary>
/// Builds the full set of external-source probes the OmniMind fans out across.
/// Every source is reached live; key-gated ones degrade to "skipped" when the
/// relevant environment variable is absent.
/// </summary>
public static class KnowledgeSources
{
    public static List<KnowledgeProbe> All(HttpClient http)
    {
        KnowledgeProbe P(string name, string category, Func<string, CancellationToken, Task<ProbeResult>> run)
            => new() { Name = name, Category = category, RunAsync = run };

        return
        [
            // ── Encyclopedic knowledge ───────────────────────────────────────────
            P("Wikipedia", "Knowledge", async (topic, ct) =>
            {
                using var wiki = new WikipediaSource(http);
                var titles = await wiki.SearchAsync(topic, 5, ct);
                if (titles.Count == 0) return ProbeResult.Empty;
                var extract = await wiki.GetExtractAsync(titles[0], ct);
                var chunks = new List<string>();
                if (!string.IsNullOrWhiteSpace(extract)) chunks.Add(Clip(extract, 8000));
                foreach (var t in titles.Skip(1)) chunks.Add(t);
                return ProbeResult.Ok(chunks, $"{titles.Count} pages · “{titles[0]}”");
            }),

            P("Wikidata", "Knowledge", async (topic, ct) =>
            {
                using var wd = new WikidataSource(http);
                var hits = await wd.SearchAsync(topic, "en", 15, ct);
                if (hits.Count == 0) return ProbeResult.Empty;
                var chunks = hits.Select(h => $"{h.Label} — {h.Description}")
                                 .Where(s => s.Trim().Length > 3).ToList();
                return ProbeResult.Ok(chunks, $"{hits.Count} entities");
            }),

            P("ConceptNet", "Knowledge", async (topic, ct) =>
            {
                using var cn = new ConceptNetSource(http);
                string term = topic.Trim().Replace(' ', '_').ToLowerInvariant();
                var edges = await cn.GetEdgesAsync(term, "en", 40, ct);
                if (edges.Count == 0) return ProbeResult.Empty;
                var chunks = edges.Where(e => e.Start.Length > 0 && e.End.Length > 0)
                                  .Select(e => $"{e.Start} {e.Relation} {e.End}").ToList();
                return ProbeResult.Ok(chunks, $"{edges.Count} relations");
            }),

            P("DBpedia", "Knowledge", async (topic, ct) =>
            {
                using var sp = SparqlSource.DBpedia(http);
                string label = Capitalize(topic);
                string q = "SELECT ?abstract WHERE { ?s rdfs:label \"" + label.Replace("\"", "") +
                           "\"@en ; dbo:abstract ?abstract . FILTER(lang(?abstract)=\"en\") } LIMIT 1";
                var res = await sp.QueryAsync(q, ct);
                var chunks = new List<string>();
                foreach (var row in res.Rows)
                    if (row.TryGetValue("abstract", out var v) && !string.IsNullOrWhiteSpace(v.Value))
                        chunks.Add(Clip(v.Value, 4000));
                return chunks.Count == 0 ? ProbeResult.Empty : ProbeResult.Ok(chunks, "dbo:abstract (SPARQL)");
            }),

            // ── Science & scholarship ────────────────────────────────────────────
            P("arXiv", "Science", async (topic, ct) =>
            {
                using var arx = new ArxivSource(http);
                var entries = await arx.SearchAsync(topic, 8, ct);
                if (entries.Count == 0) return ProbeResult.Empty;
                var chunks = entries.Select(e => $"{e.Title}. {e.Summary}").ToList();
                return ProbeResult.Ok(chunks, $"{entries.Count} papers");
            }),

            P("PubMed", "Science", async (topic, ct) =>
            {
                using var pm = new PubMedSource(http);
                var ids = await pm.SearchIdsAsync(topic, 12, ct);
                if (ids.Count == 0) return ProbeResult.Empty;
                var arts = await pm.GetSummariesAsync(ids, ct);
                var chunks = arts.Where(a => !string.IsNullOrWhiteSpace(a.Title)).Select(a => a.Title!).ToList();
                return chunks.Count == 0 ? ProbeResult.Empty : ProbeResult.Ok(chunks, $"{arts.Count} articles");
            }),

            P("OpenAlex", "Science", async (topic, ct) =>
            {
                using var oa = new OpenAlexSource(http);
                var works = await oa.SearchWorksAsync(topic, 15, ct);
                if (works.Count == 0) return ProbeResult.Empty;
                var chunks = works.Where(w => !string.IsNullOrWhiteSpace(w.Title)).Select(w => w.Title).ToList();
                return ProbeResult.Ok(chunks, $"{works.Count} works");
            }),

            P("Unpaywall", "Science", async (topic, ct) =>
            {
                string? email = Environment.GetEnvironmentVariable("UNPAYWALL_EMAIL");
                if (string.IsNullOrWhiteSpace(email)) return ProbeResult.Skip("set UNPAYWALL_EMAIL");
                using var oa = new OpenAlexSource(http);
                var works = await oa.SearchWorksAsync(topic, 10, ct);
                string? doi = works.Select(w => w.Doi).FirstOrDefault(d => !string.IsNullOrWhiteSpace(d));
                if (doi is null) return ProbeResult.Empty;
                doi = doi.Replace("https://doi.org/", "");
                using var up = new UnpaywallSource(email, http);
                var rec = await up.GetAsync(doi, ct);
                return ProbeResult.Ok([rec.Title ?? doi], rec.IsOpenAccess ? "open access" : "closed");
            }),

            P("ScienceDirect", "Science", async (topic, ct) =>
            {
                string? key = Environment.GetEnvironmentVariable("ELSEVIER_API_KEY");
                if (string.IsNullOrWhiteSpace(key)) return ProbeResult.Skip("set ELSEVIER_API_KEY");
                using var sd = new ScienceDirectSource(key, http);
                var arts = await sd.SearchAsync(topic, 15, ct);
                var chunks = arts.Where(a => !string.IsNullOrWhiteSpace(a.Title)).Select(a => a.Title!).ToList();
                return chunks.Count == 0 ? ProbeResult.Empty : ProbeResult.Ok(chunks, $"{arts.Count} articles");
            }),

            P("Hugging Face", "Science", async (topic, ct) =>
            {
                using var hf = new HuggingFaceDatasets(http);
                var rows = await hf.GetRowsAsync("stanfordnlp/imdb", "plain_text", "train", 0, 5, ct);
                var chunks = rows
                    .Select(r => r.TryGetValue("text", out var t)
                        ? t?.ToString() ?? ""
                        : string.Join(" ", r.Values.Select(v => v?.ToString())))
                    .Where(s => !string.IsNullOrWhiteSpace(s)).Select(s => Clip(s, 1500)).ToList();
                return chunks.Count == 0 ? ProbeResult.Empty : ProbeResult.Ok(chunks, "dataset rows (imdb)");
            }),

            // ── Social & community ───────────────────────────────────────────────
            P("Hacker News", "Social", async (topic, ct) =>
            {
                using var hn = new HackerNewsSource(http);
                var ids = await hn.GetTopStoryIdsAsync(ct);
                var take = ids.Take(15).ToList();
                var items = await Task.WhenAll(take.Select(id => SafeItem(hn, id, ct)));
                var titles = items.Where(i => i?.Title is not null).Select(i => i!.Title!).ToList();
                if (titles.Count == 0) return ProbeResult.Empty;
                var toks = Tokens(topic);
                var rel = titles.Where(t => toks.Any(k => t.Contains(k, StringComparison.OrdinalIgnoreCase))).ToList();
                var chunks = rel.Count > 0 ? rel : titles.Take(8).ToList();
                return ProbeResult.Ok(chunks, rel.Count > 0 ? $"{rel.Count} on-topic" : "top stories");
            }),

            P("Reddit", "Social", async (topic, ct) =>
            {
                using var rd = new RedditSource(http);
                string sub = Subreddit(topic);
                var posts = await rd.GetPostsAsync(sub, "hot", 15, ct);
                if (posts.Count == 0) return ProbeResult.Empty;
                var chunks = posts
                    .Select(p => string.IsNullOrWhiteSpace(p.SelfText) ? p.Title ?? "" : $"{p.Title}. {p.SelfText}")
                    .Where(s => !string.IsNullOrWhiteSpace(s)).Select(s => Clip(s, 1200)).ToList();
                return chunks.Count == 0 ? ProbeResult.Empty : ProbeResult.Ok(chunks, $"r/{sub} · {posts.Count}");
            }),

            P("Mastodon", "Social", async (topic, ct) =>
            {
                using var ms = new MastodonSource(http);
                var statuses = await ms.GetPublicTimelineAsync("mastodon.social", 40, ct);
                var toks = Tokens(topic);
                var rel = statuses.Where(s => toks.Any(k => s.Text.Contains(k, StringComparison.OrdinalIgnoreCase)))
                                  .Select(s => s.Text).ToList();
                var chunks = (rel.Count > 0 ? rel : statuses.Take(10).Select(s => s.Text))
                             .Where(t => !string.IsNullOrWhiteSpace(t)).ToList();
                return chunks.Count == 0 ? ProbeResult.Empty
                    : ProbeResult.Ok(chunks, rel.Count > 0 ? $"{rel.Count} on-topic" : "fediverse pulse");
            }),

            // ── Code ─────────────────────────────────────────────────────────────
            P("GitHub", "Code", async (topic, ct) =>
            {
                var gh = new GitHubSource(token: Environment.GetEnvironmentVariable("GITHUB_TOKEN"));
                var repos = await gh.SearchRepositoriesAsync(topic, ct);
                if (repos.Count == 0) return ProbeResult.Empty;
                var chunks = repos.Take(20)
                    .Select(r => string.IsNullOrWhiteSpace(r.Description) ? r.FullName : $"{r.FullName}: {r.Description}")
                    .ToList();
                return ProbeResult.Ok(chunks, $"{repos.Count} repos");
            }),

            P("Stack Overflow", "Code", async (topic, ct) =>
            {
                using var se = new StackExchangeSource();   // owns a gzip-enabled client
                var qs = await se.SearchAsync(topic, "stackoverflow", ct);
                if (qs.Count == 0) return ProbeResult.Empty;
                var chunks = qs.Where(q => q.Title is not null).Select(q => q.Title!).ToList();
                return ProbeResult.Ok(chunks, $"{qs.Count} questions");
            }),

            // ── Open web & media ─────────────────────────────────────────────────
            P("News (RSS)", "Web", async (topic, ct) =>
            {
                string url = $"https://news.google.com/rss/search?q={Uri.EscapeDataString(topic)}&hl=en-US&gl=US&ceid=US:en";
                var feed = await RssReader.Instance.ReadAsync(url, ct);
                var chunks = feed.Items
                    .Select(i => string.IsNullOrWhiteSpace(i.Summary) ? i.Title ?? "" : $"{i.Title}. {StripTags(i.Summary!)}")
                    .Where(s => !string.IsNullOrWhiteSpace(s)).ToList();
                return chunks.Count == 0 ? ProbeResult.Empty : ProbeResult.Ok(chunks, $"{chunks.Count} headlines");
            }),

            P("Web (AngleSharp)", "Web", async (topic, ct) =>
            {
                using var web = new HtmlSource(http);
                string url = $"https://lite.duckduckgo.com/lite/?q={Uri.EscapeDataString(topic)}";
                var content = await web.ReadAsync(url, ct);
                if (string.IsNullOrWhiteSpace(content.Text)) return ProbeResult.Empty;
                return ProbeResult.Ok([Clip(content.Text, 6000)], $"{content.Links.Count} links · {content.Text.Length} chars");
            }),

            P("YouTube", "Web", async (topic, ct) =>
            {
                var yt = new YouTubeSource();
                var titles = new List<string>();
                await foreach (var r in yt.SearchVideosAsync(topic, ct))
                {
                    titles.Add(r.Title);
                    if (titles.Count >= 12) break;
                }
                return titles.Count == 0 ? ProbeResult.Empty : ProbeResult.Ok(titles, $"{titles.Count} videos");
            }),

            // ── Live signals (numeric / geographic → one fact each) ──────────────
            P("Open-Meteo", "Signal", async (topic, ct) =>
            {
                var geo = await Geocode(http, topic, ct);
                if (geo is null) return new ProbeResult(ProbeState.Empty, 0, Array.Empty<string>(), "not a place");
                using var w = new WeatherSource(http);
                var series = await w.GetHourlyAsync(geo.Value.lat, geo.Value.lon, ["temperature_2m"], ct);
                float t = series.Values.Buffer.Length > 0 ? series.Values.Buffer[0] : float.NaN;
                return new ProbeResult(ProbeState.Ok, series.Times.Count,
                    [$"In {geo.Value.name}, the temperature is around {t:F1} degrees Celsius."], $"{geo.Value.name} {t:F1}°C");
            }),

            P("Coinbase", "Signal", async (topic, ct) =>
            {
                using var cb = new CoinbaseSource(http);
                string pair = Coin(topic);
                decimal price = await cb.GetSpotPriceAsync(pair, ct);
                return new ProbeResult(ProbeState.Ok, 1,
                    [$"The {pair} spot price is {price:N2} USD."], $"{pair} {price:N0}");
            }),

            P("Stooq", "Signal", async (topic, ct) =>
            {
                using var st = new StooqSource(http);
                string sym = Ticker(topic);
                var series = await st.GetDailyAsync(sym, ct);
                int n = series.Dates.Count;
                float close = series.Values.Buffer[(n - 1) * 5 + 3];
                return new ProbeResult(ProbeState.Ok, n,
                    [$"{sym} last closed at {close:N2} on {series.Dates[n - 1]:yyyy-MM-dd}."], $"{sym} {close:N2}");
            }),

            P("World Bank", "Signal", async (topic, ct) =>
            {
                using var wb = new WorldBankSource(http);
                var pts = await wb.GetIndicatorAsync("WLD", "SP.POP.TOTL", ct);
                var latest = pts.Where(p => p.Value.HasValue).OrderByDescending(p => p.Year).FirstOrDefault();
                if (latest is null) return ProbeResult.Empty;
                return new ProbeResult(ProbeState.Ok, pts.Count,
                    [$"World population in {latest.Year} was about {latest.Value:N0} people."], $"pop {latest.Value:N0}");
            }),

            P("OpenStreetMap", "Signal", async (topic, ct) =>
            {
                using var ov = new OverpassSource(http);
                string ql = "[out:json][timeout:20];node[\"name\"=\"" + topic.Replace("\"", "") + "\"];out 5;";
                var els = await ov.QueryAsync(ql, ct);
                if (els.Count == 0) return ProbeResult.Empty;
                var chunks = els.Where(e => e.Tags.Count > 0)
                    .Select(e => $"{topic}: " + string.Join(", ", e.Tags.Take(4).Select(kv => $"{kv.Key}={kv.Value}")))
                    .ToList();
                return new ProbeResult(ProbeState.Ok, els.Count,
                    chunks.Count > 0 ? chunks : [$"{els.Count} OSM elements named {topic}"], $"{els.Count} elements");
            }),
        ];
    }

    // ── Per-source helpers ──────────────────────────────────────────────────────

    private static async Task<HackerNewsItem?> SafeItem(HackerNewsSource hn, long id, CancellationToken ct)
    {
        try { return await hn.GetItemAsync(id, ct); }
        catch { return null; }
    }

    private static async Task<(double lat, double lon, string name)?> Geocode(HttpClient http, string topic, CancellationToken ct)
    {
        string url = $"https://geocoding-api.open-meteo.com/v1/search?name={Uri.EscapeDataString(topic)}&count=1";
        string json = await http.GetStringAsync(url, ct).ConfigureAwait(false);
        using var doc = JsonDocument.Parse(json);
        if (doc.RootElement.TryGetProperty("results", out var r) && r.ValueKind == JsonValueKind.Array && r.GetArrayLength() > 0)
        {
            var e = r[0];
            return (e.GetProperty("latitude").GetDouble(),
                    e.GetProperty("longitude").GetDouble(),
                    e.TryGetProperty("name", out var nm) ? nm.GetString() ?? topic : topic);
        }
        return null;
    }

    private static string[] Tokens(string topic)
        => topic.ToLowerInvariant()
                .Split(new[] { ' ', '-', '_', ',', '.' }, StringSplitOptions.RemoveEmptyEntries)
                .Where(t => t.Length >= 3).ToArray();

    private static string Clip(string s, int max) => s.Length <= max ? s : s[..max];

    private static string StripTags(string html) => Regex.Replace(html, "<[^>]+>", " ").Replace("&nbsp;", " ").Trim();

    private static string Capitalize(string s)
        => s.Length == 0 ? s : char.ToUpperInvariant(s[0]) + s[1..];

    private static string Subreddit(string topic)
        => new string(topic.Where(char.IsLetterOrDigit).ToArray()) is { Length: > 0 } s ? s : "all";

    private static string Coin(string topic)
    {
        string t = topic.ToLowerInvariant();
        if (t.Contains("ethereum") || t.Contains("ether")) return "ETH-USD";
        if (t.Contains("solana")) return "SOL-USD";
        if (t.Contains("dogecoin") || t.Contains("doge")) return "DOGE-USD";
        if (t.Contains("cardano")) return "ADA-USD";
        return "BTC-USD";
    }

    private static string Ticker(string topic)
    {
        string t = topic.ToLowerInvariant();
        if (t.Contains("apple")) return "aapl.us";
        if (t.Contains("microsoft")) return "msft.us";
        if (t.Contains("tesla")) return "tsla.us";
        if (t.Contains("nvidia")) return "nvda.us";
        if (t.Contains("amazon")) return "amzn.us";
        if (t.Contains("google") || t.Contains("alphabet")) return "googl.us";
        if (t.Contains("meta") || t.Contains("facebook")) return "meta.us";
        return "^spx";
    }
}

/// <summary>Light cross-source concept extraction: most frequent meaningful words
/// across the fused corpus, excluding stopwords and the topic itself.</summary>
public static class Insights
{
    private static readonly HashSet<string> Stop = new(StringComparer.OrdinalIgnoreCase)
    {
        "about","there","their","which","these","those","other","would","could","should","because",
        "while","where","being","using","based","first","https","http","html","said","also","into",
        "from","that","this","with","have","what","when","they","them","then","than","were","will",
        "your","you","not","are","was","has","had","but","for","and","the","its","may","can","such",
        "more","most","some","many","over","under","between","after","before","through","during",
    };

    public static IReadOnlyList<string> TopTerms(string corpus, string topic, int n)
    {
        if (string.IsNullOrWhiteSpace(corpus)) return Array.Empty<string>();
        var topicTokens = new HashSet<string>(
            topic.ToLowerInvariant().Split(' ', StringSplitOptions.RemoveEmptyEntries));

        var counts = new Dictionary<string, int>();
        foreach (var raw in Regex.Split(corpus.ToLowerInvariant(), "[^a-z]+"))
        {
            if (raw.Length < 5 || Stop.Contains(raw) || topicTokens.Contains(raw)) continue;
            counts[raw] = counts.GetValueOrDefault(raw) + 1;
        }
        return counts.OrderByDescending(kv => kv.Value).Take(n)
                     .Select(kv => $"{kv.Key} ({kv.Value})").ToList();
    }
}
