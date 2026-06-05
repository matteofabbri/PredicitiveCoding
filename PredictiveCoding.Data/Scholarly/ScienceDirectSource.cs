using System.Text.Json;

namespace PredictiveCoding.Data.Scholarly;

/// <summary>An article record from ScienceDirect / Elsevier.</summary>
public sealed record ScienceDirectArticle(
    string? Title,
    string? Authors,
    string? Doi,
    string? PublicationName,
    string? CoverDate,
    string? Url);

/// <summary>
/// Data source for Elsevier ScienceDirect via its official Content API
/// (api.elsevier.com). A valid Elsevier API key is required; JSON parsing is
/// static and runs offline. Full-text access depends on the caller's
/// entitlements — this exposes search and article metadata/abstracts.
/// </summary>
public sealed class ScienceDirectSource : IDisposable
{
    private const string BaseUrl = "https://api.elsevier.com/content/";

    private readonly HttpClient _http;
    private readonly bool _ownsHttp;
    private readonly string _apiKey;

    public ScienceDirectSource(string apiKey, HttpClient? httpClient = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(apiKey);
        _apiKey = apiKey;
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Searches ScienceDirect.</summary>
    public async Task<IReadOnlyList<ScienceDirectArticle>> SearchAsync(
        string query, int count = 25, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        if (count <= 0) throw new ArgumentOutOfRangeException(nameof(count));
        string url = $"{BaseUrl}search/sciencedirect?count={count}&query={Uri.EscapeDataString(query)}";
        return ParseSearch(await GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Retrieves a single article's metadata/abstract by DOI.</summary>
    public async Task<ScienceDirectArticle> GetByDoiAsync(string doi, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(doi);
        string url = $"{BaseUrl}article/doi/{doi}";
        return ParseArticle(await GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    private async Task<string> GetStringAsync(string url, CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        request.Headers.Add("X-ELS-APIKey", _apiKey);
        request.Headers.Accept.ParseAdd("application/json");
        using HttpResponseMessage response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Parses a ScienceDirect search response (offline).</summary>
    public static IReadOnlyList<ScienceDirectArticle> ParseSearch(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);

        var articles = new List<ScienceDirectArticle>();
        if (document.RootElement.TryGetProperty("search-results", out var results) &&
            results.TryGetProperty("entry", out var entries) && entries.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement entry in entries.EnumerateArray())
                articles.Add(MapCoreData(entry));
        }
        return articles;
    }

    /// <summary>Parses a ScienceDirect article-retrieval response (offline).</summary>
    public static ScienceDirectArticle ParseArticle(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        JsonElement core = document.RootElement;
        if (core.TryGetProperty("full-text-retrieval-response", out var response) &&
            response.TryGetProperty("coredata", out var coredata))
            core = coredata;
        return MapCoreData(core);
    }

    private static ScienceDirectArticle MapCoreData(JsonElement element)
        => new(
            Title: GetString(element, "dc:title"),
            Authors: Creators(element),
            Doi: GetString(element, "prism:doi"),
            PublicationName: GetString(element, "prism:publicationName"),
            CoverDate: GetString(element, "prism:coverDate"),
            Url: GetString(element, "prism:url"));

    private static string? Creators(JsonElement element)
    {
        if (!element.TryGetProperty("dc:creator", out var creator))
            return null;
        switch (creator.ValueKind)
        {
            case JsonValueKind.String:
                return creator.GetString();
            case JsonValueKind.Array:
                var names = new List<string>();
                foreach (var item in creator.EnumerateArray())
                {
                    string? name = item.ValueKind == JsonValueKind.String
                        ? item.GetString()
                        : GetString(item, "$") ?? GetString(item, "authname");
                    if (!string.IsNullOrWhiteSpace(name))
                        names.Add(name);
                }
                return names.Count > 0 ? string.Join(", ", names) : null;
            case JsonValueKind.Object:
                return GetString(creator, "$") ?? GetString(creator, "authname");
            default:
                return null;
        }
    }

    private static string? GetString(JsonElement parent, string name)
        => parent.ValueKind == JsonValueKind.Object && parent.TryGetProperty(name, out var element) &&
           element.ValueKind == JsonValueKind.String
            ? element.GetString()
            : null;
}
