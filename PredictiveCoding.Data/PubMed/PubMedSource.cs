using System.Text.Json;

namespace PredictiveCoding.Data.PubMed;

/// <summary>A PubMed article summary.</summary>
public sealed record PubMedArticle(string Id, string? Title, string? PubDate, string? Journal, IReadOnlyList<string> Authors);

/// <summary>
/// Biomedical-literature source backed by the NCBI E-utilities (esearch +
/// esummary). JSON parsing is static and runs offline.
/// </summary>
public sealed class PubMedSource : IDisposable
{
    private const string BaseUrl = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/";

    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public PubMedSource(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Searches PubMed and returns matching article ids.</summary>
    public async Task<IReadOnlyList<string>> SearchIdsAsync(string query, int retMax = 20, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        if (retMax <= 0) throw new ArgumentOutOfRangeException(nameof(retMax));
        string url = $"{BaseUrl}esearch.fcgi?db=pubmed&retmode=json&retmax={retMax}&term={Uri.EscapeDataString(query)}";
        return ParseIds(await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Fetches article summaries for the given ids.</summary>
    public async Task<IReadOnlyList<PubMedArticle>> GetSummariesAsync(IEnumerable<string> ids, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(ids);
        string joined = string.Join(',', ids);
        if (string.IsNullOrWhiteSpace(joined)) throw new ArgumentException("At least one id is required.", nameof(ids));
        string url = $"{BaseUrl}esummary.fcgi?db=pubmed&retmode=json&id={Uri.EscapeDataString(joined)}";
        return ParseSummaries(await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses an esearch response (offline).</summary>
    public static IReadOnlyList<string> ParseIds(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        var ids = new List<string>();
        if (document.RootElement.TryGetProperty("esearchresult", out var result) &&
            result.TryGetProperty("idlist", out var idList) && idList.ValueKind == JsonValueKind.Array)
            foreach (var id in idList.EnumerateArray())
                if (id.ValueKind == JsonValueKind.String)
                    ids.Add(id.GetString()!);
        return ids;
    }

    /// <summary>Parses an esummary response (offline).</summary>
    public static IReadOnlyList<PubMedArticle> ParseSummaries(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        var articles = new List<PubMedArticle>();
        if (!document.RootElement.TryGetProperty("result", out var result) || result.ValueKind != JsonValueKind.Object)
            return articles;

        if (!result.TryGetProperty("uids", out var uids) || uids.ValueKind != JsonValueKind.Array)
            return articles;

        foreach (var uidElement in uids.EnumerateArray())
        {
            string? uid = uidElement.GetString();
            if (uid is null || !result.TryGetProperty(uid, out var article))
                continue;

            var authors = new List<string>();
            if (article.TryGetProperty("authors", out var authorArray) && authorArray.ValueKind == JsonValueKind.Array)
                foreach (var author in authorArray.EnumerateArray())
                    if (author.TryGetProperty("name", out var name) && name.ValueKind == JsonValueKind.String)
                        authors.Add(name.GetString()!);

            articles.Add(new PubMedArticle(
                uid,
                GetString(article, "title"),
                GetString(article, "pubdate"),
                GetString(article, "fulljournalname"),
                authors));
        }
        return articles;
    }

    private static string? GetString(JsonElement parent, string name)
        => parent.TryGetProperty(name, out var element) && element.ValueKind == JsonValueKind.String
            ? element.GetString()
            : null;
}
