using PredictiveCoding.Data.Text;

namespace PredictiveCoding.Data.Web;

/// <summary>Crawl behaviour knobs (all honest defaults: polite, same-host, boilerplate-filtered).</summary>
public sealed record CrawlOptions
{
    /// <summary>Stop after this many fetched pages.</summary>
    public int MaxPages { get; init; } = 100;

    /// <summary>Follow links only on the seed hosts (true) or anywhere (false).</summary>
    public bool SameHostOnly { get; init; } = true;

    /// <summary>Minimum pause between two fetches to the SAME host.</summary>
    public TimeSpan PerHostDelay { get; init; } = TimeSpan.FromSeconds(1);

    /// <summary>Blocks with fewer words than this are dropped (nav/menu boilerplate).</summary>
    public int MinBlockWords { get; init; } = 5;

    /// <summary>Pages whose surviving blocks total fewer characters are not emitted (links still followed).</summary>
    public int MinPageChars { get; init; } = 200;

    /// <summary>
    /// Allowed language labels as the classifier names them (e.g. "English",
    /// "Italian"); null accepts everything.
    /// </summary>
    public IReadOnlyList<string>? Languages { get; init; }

    /// <summary>
    /// text → language label, used when <see cref="Languages"/> is set. Null
    /// uses the library's embedded detector — which today is weak on short web
    /// text (its seed corpus is tiny), so a real crawl should inject a stronger
    /// classifier until the embedded profile is retrained.
    /// </summary>
    public Func<string, string>? LanguageClassifier { get; init; }

    /// <summary>Cap on the HTML size fetched per page.</summary>
    public long MaxHtmlBytes { get; init; } = 2 * 1024 * 1024;

    public string UserAgent { get; init; } = "PredictiveCoding.Crawler/1.0 (research corpus builder)";
}

/// <summary>One kept page: cleaned content blocks plus what the crawler learned about it.</summary>
public sealed record CrawledPage(string Url, string? Title, IReadOnlyList<string> Blocks, string? Language);

/// <summary>What a crawl did, in numbers.</summary>
public sealed record CrawlReport(int PagesFetched, int PagesKept, int PagesSkippedLanguage,
                                 int PagesSkippedThin, int PagesFailed, long BlocksEmitted);

// ─────────────────────────────────────────────────────────────────────────────
// WebCrawler  —  the live crawler that SEDIMENTS a corpus (ROADMAP-Scale C1).
//
// BFS over a link frontier from the seed URLs: fetch politely (per-host delay,
// honest user agent, HTML-only, size-capped), extract block-level text through
// HtmlSource.ParseBlocks (chrome stripped, short blocks dropped — that is the
// boilerplate filter), optionally keep only configured languages
// (PredictiveCoding.Data's own detector), and hand every kept page to the
// caller — typically a ShardedCorpusWriter (build the corpus) and, once a
// bootstrapped model exists, a ContinualLearner (C4: the stream, forever).
// Fetching is never gated by what the consumer does with the page.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class WebCrawler : IDisposable
{
    private static readonly string[] SkipExtensions =
        [".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico", ".css", ".js",
         ".pdf", ".zip", ".gz", ".xz", ".rar", ".7z", ".mp3", ".mp4", ".avi", ".mkv",
         ".woff", ".woff2", ".ttf", ".exe", ".dmg", ".iso"];

    private readonly CrawlOptions _opts;
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;
    private readonly HtmlSource _parser = new(new HttpClient());   // used as a parser only
    // Seeded: the corpus a crawl keeps must not depend on RNG luck.
    private readonly Lazy<Func<string, string>> _classify;

    public WebCrawler(CrawlOptions? options = null, HttpClient? httpClient = null)
    {
        _opts = options ?? new CrawlOptions();
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
        _classify = new(() => _opts.LanguageClassifier
                              ?? LanguageDetector.CreateDefault(seed: 42).Detect);
    }

    /// <summary>
    /// Breadth-first crawl from <paramref name="seeds"/>: every kept page goes to
    /// <paramref name="onPage"/> as it is fetched. Returns the tally.
    /// </summary>
    public async Task<CrawlReport> CrawlAsync(IEnumerable<string> seeds, Action<CrawledPage>? onPage,
                                              CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(seeds);

        var frontier = new Queue<Uri>();
        var visited = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var allowedHosts = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        var lastFetch = new Dictionary<string, DateTime>(StringComparer.OrdinalIgnoreCase);

        foreach (string seed in seeds)
        {
            var uri = new Uri(seed, UriKind.Absolute);
            allowedHosts.Add(uri.Host);
            if (visited.Add(Normalize(uri))) frontier.Enqueue(uri);
        }

        int fetched = 0, kept = 0, skippedLanguage = 0, skippedThin = 0, failed = 0;
        long blocksEmitted = 0;

        while (frontier.Count > 0 && fetched < _opts.MaxPages && !cancellationToken.IsCancellationRequested)
        {
            var uri = frontier.Dequeue();

            // Politeness: wait out the per-host delay.
            if (lastFetch.TryGetValue(uri.Host, out var last))
            {
                var wait = last + _opts.PerHostDelay - DateTime.UtcNow;
                if (wait > TimeSpan.Zero) await Task.Delay(wait, cancellationToken).ConfigureAwait(false);
            }

            string html;
            try
            {
                lastFetch[uri.Host] = DateTime.UtcNow;
                fetched++;
                html = await FetchHtml(uri, cancellationToken).ConfigureAwait(false);
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested) { break; }
            catch { failed++; continue; }
            if (html.Length == 0) { failed++; continue; }          // not HTML / too big

            // Links first: the frontier grows even off pages we do not keep.
            var content = _parser.Parse(html);
            foreach (string href in content.Links)
            {
                if (!Uri.TryCreate(uri, href, out var link)) continue;
                if (link.Scheme != Uri.UriSchemeHttp && link.Scheme != Uri.UriSchemeHttps) continue;
                if (_opts.SameHostOnly && !allowedHosts.Contains(link.Host)) continue;
                if (HasSkippedExtension(link)) continue;
                if (visited.Add(Normalize(link))) frontier.Enqueue(link);
            }

            // Content: block extraction + boilerplate word filter.
            var blocks = new List<string>();
            foreach (string block in _parser.ParseBlocks(html))
                if (CountWords(block) >= _opts.MinBlockWords)
                    blocks.Add(block);

            int totalChars = blocks.Sum(b => b.Length);
            if (totalChars < _opts.MinPageChars) { skippedThin++; continue; }

            string? language = null;
            if (_opts.Languages is { Count: > 0 })
            {
                string sample = string.Join(' ', blocks);
                language = _classify.Value(sample.Length <= 1000 ? sample : sample[..1000]);
                if (!_opts.Languages.Contains(language, StringComparer.OrdinalIgnoreCase))
                {
                    skippedLanguage++;
                    continue;
                }
            }

            kept++;
            blocksEmitted += blocks.Count;
            onPage?.Invoke(new CrawledPage(uri.ToString(), content.Title, blocks, language));
        }

        return new CrawlReport(fetched, kept, skippedLanguage, skippedThin, failed, blocksEmitted);
    }

    private async Task<string> FetchHtml(Uri uri, CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, uri);
        request.Headers.UserAgent.ParseAdd(_opts.UserAgent);

        using var response = await _http.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
                                        .ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        string? mediaType = response.Content.Headers.ContentType?.MediaType;
        if (mediaType is not null && !mediaType.Contains("html", StringComparison.OrdinalIgnoreCase))
            return string.Empty;
        if (response.Content.Headers.ContentLength is > 0 and var len && len > _opts.MaxHtmlBytes)
            return string.Empty;

        return await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
    }

    private static bool HasSkippedExtension(Uri uri)
    {
        string path = uri.AbsolutePath;
        foreach (string ext in SkipExtensions)
            if (path.EndsWith(ext, StringComparison.OrdinalIgnoreCase)) return true;
        return false;
    }

    private static string Normalize(Uri uri) => uri.GetLeftPart(UriPartial.Query);

    private static int CountWords(string text)
    {
        int words = 0; bool inWord = false;
        foreach (char c in text)
        {
            if (char.IsWhiteSpace(c)) inWord = false;
            else if (!inWord) { inWord = true; words++; }
        }
        return words;
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
        _parser.Dispose();
    }
}
