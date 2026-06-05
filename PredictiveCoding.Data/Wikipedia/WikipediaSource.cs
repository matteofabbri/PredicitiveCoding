using System.Text.Json;

namespace PredictiveCoding.Data.Wikipedia;

/// <summary>A Wikipedia page summary.</summary>
public sealed record WikipediaPage(string Title, string? Description, string? Extract, string? Url);

/// <summary>
/// Data source for Wikipedia via the REST summary endpoint and the MediaWiki
/// action API. JSON parsing is static and runs offline.
/// </summary>
public sealed class WikipediaSource : IDisposable
{
    private const string DefaultUserAgent = "PredictiveCoding.Data/1.0 (Wikipedia data source)";

    private readonly HttpClient _http;
    private readonly bool _ownsHttp;
    private readonly string _language;
    private readonly string _userAgent;

    public WikipediaSource(HttpClient? httpClient = null, string language = "en", string? userAgent = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(language);
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
        _language = language;
        _userAgent = string.IsNullOrWhiteSpace(userAgent) ? DefaultUserAgent : userAgent;
    }

    /// <summary>Fetches the short summary of a page (REST API).</summary>
    public async Task<WikipediaPage> GetSummaryAsync(string title, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(title);
        string url = $"https://{_language}.wikipedia.org/api/rest_v1/page/summary/{Uri.EscapeDataString(title)}";
        return ParseSummary(await GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Fetches the full plain-text extract of a page (action API).</summary>
    public async Task<string> GetExtractAsync(string title, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(title);
        string url = $"https://{_language}.wikipedia.org/w/api.php?action=query&format=json&prop=extracts"
                   + $"&explaintext=1&redirects=1&titles={Uri.EscapeDataString(title)}";
        return ParseExtract(await GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Searches page titles (action API).</summary>
    public async Task<IReadOnlyList<string>> SearchAsync(string query, int limit = 10, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        if (limit <= 0) throw new ArgumentOutOfRangeException(nameof(limit));
        string url = $"https://{_language}.wikipedia.org/w/api.php?action=query&format=json&list=search"
                   + $"&srlimit={limit}&srsearch={Uri.EscapeDataString(query)}";
        return ParseSearch(await GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    // ── Parsing (offline) ────────────────────────────────────────────────────

    /// <summary>Parses a REST summary response.</summary>
    public static WikipediaPage ParseSummary(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        string title = GetString(root, "title") ?? string.Empty;
        string? url = null;
        if (root.TryGetProperty("content_urls", out var urls) &&
            urls.TryGetProperty("desktop", out var desktop))
            url = GetString(desktop, "page");

        return new WikipediaPage(title, GetString(root, "description"), GetString(root, "extract"), url);
    }

    /// <summary>Parses the extract of the first page from an action-API response.</summary>
    public static string ParseExtract(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        if (document.RootElement.TryGetProperty("query", out var query) &&
            query.TryGetProperty("pages", out var pages) && pages.ValueKind == JsonValueKind.Object)
        {
            foreach (var page in pages.EnumerateObject())
                if (page.Value.TryGetProperty("extract", out var extract) && extract.ValueKind == JsonValueKind.String)
                    return extract.GetString() ?? string.Empty;
        }
        return string.Empty;
    }

    /// <summary>Parses search-result titles from an action-API response.</summary>
    public static IReadOnlyList<string> ParseSearch(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        var titles = new List<string>();
        if (document.RootElement.TryGetProperty("query", out var query) &&
            query.TryGetProperty("search", out var search) && search.ValueKind == JsonValueKind.Array)
        {
            foreach (var hit in search.EnumerateArray())
            {
                string? title = GetString(hit, "title");
                if (title is not null) titles.Add(title);
            }
        }
        return titles;
    }

    private async Task<string> GetStringAsync(string url, CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        request.Headers.UserAgent.ParseAdd(_userAgent);
        using HttpResponseMessage response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
    }

    private static string? GetString(JsonElement parent, string name)
        => parent.ValueKind == JsonValueKind.Object && parent.TryGetProperty(name, out var element) &&
           element.ValueKind == JsonValueKind.String
            ? element.GetString()
            : null;
}
