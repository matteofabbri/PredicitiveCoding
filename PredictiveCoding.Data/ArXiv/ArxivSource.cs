using System.Globalization;
using System.Xml.Linq;

namespace PredictiveCoding.Data.ArXiv;

/// <summary>A paper entry from the arXiv API.</summary>
public sealed record ArxivEntry(
    string Id,
    string Title,
    string Summary,
    IReadOnlyList<string> Authors,
    DateTimeOffset? Published,
    string? PdfUrl,
    IReadOnlyList<string> Categories);

/// <summary>
/// Data source for arXiv via its Atom query API. The Atom parser is static and
/// runs offline.
/// </summary>
public sealed class ArxivSource : IDisposable
{
    private static readonly XNamespace Atom = "http://www.w3.org/2005/Atom";

    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public ArxivSource(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
        // arXiv's API guidelines ask clients to identify themselves; anonymous
        // requests are throttled aggressively.
        if (!_http.DefaultRequestHeaders.UserAgent.Any())
            _http.DefaultRequestHeaders.UserAgent.ParseAdd("PredictiveCoding/1.0");
    }

    /// <summary>Searches arXiv and returns the matching entries.</summary>
    public async Task<IReadOnlyList<ArxivEntry>> SearchAsync(
        string query, int maxResults = 10, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        if (maxResults <= 0) throw new ArgumentOutOfRangeException(nameof(maxResults));

        // https directly: the http endpoint answers 301 and some HttpClient
        // configurations (AllowAutoRedirect = false, strict proxies) never land.
        string url = $"https://export.arxiv.org/api/query?search_query={Uri.EscapeDataString(query)}"
                   + $"&start=0&max_results={maxResults}";
        string xml = await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false);
        return ParseFeed(xml);
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses an arXiv Atom feed (offline).</summary>
    public static IReadOnlyList<ArxivEntry> ParseFeed(string atomXml)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(atomXml);
        XDocument document = XDocument.Parse(atomXml);
        XElement? feed = document.Root;
        if (feed is null)
            return [];

        var entries = new List<ArxivEntry>();
        foreach (XElement entry in feed.Elements(Atom + "entry"))
        {
            var authors = entry.Elements(Atom + "author")
                .Select(a => a.Element(Atom + "name")?.Value?.Trim())
                .Where(n => !string.IsNullOrEmpty(n))
                .Select(n => n!)
                .ToList();

            string? pdfUrl = entry.Elements(Atom + "link")
                .FirstOrDefault(l => (string?)l.Attribute("title") == "pdf"
                                  || (string?)l.Attribute("type") == "application/pdf")
                ?.Attribute("href")?.Value;

            var categories = entry.Elements(Atom + "category")
                .Select(c => (string?)c.Attribute("term"))
                .Where(t => !string.IsNullOrEmpty(t))
                .Select(t => t!)
                .ToList();

            DateTimeOffset? published = null;
            if (DateTimeOffset.TryParse((string?)entry.Element(Atom + "published"),
                    CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var parsed))
                published = parsed;

            entries.Add(new ArxivEntry(
                Id: entry.Element(Atom + "id")?.Value?.Trim() ?? string.Empty,
                Title: Normalize(entry.Element(Atom + "title")?.Value),
                Summary: Normalize(entry.Element(Atom + "summary")?.Value),
                Authors: authors,
                Published: published,
                PdfUrl: pdfUrl,
                Categories: categories));
        }
        return entries;
    }

    private static string Normalize(string? text)
        => string.IsNullOrWhiteSpace(text)
            ? string.Empty
            : string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
}
