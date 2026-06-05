using System.Text.Json;

namespace PredictiveCoding.Data.Scholarly;

/// <summary>A scholarly work from OpenAlex.</summary>
public sealed record ScholarlyWork(string Id, string Title, int? Year, string? Doi, IReadOnlyList<string> Authors);

/// <summary>
/// Scholarly-metadata source backed by the OpenAlex API. JSON parsing is static
/// and runs offline.
/// </summary>
public sealed class OpenAlexSource : IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public OpenAlexSource(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Full-text searches works.</summary>
    public async Task<IReadOnlyList<ScholarlyWork>> SearchWorksAsync(
        string search, int perPage = 10, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(search);
        if (perPage <= 0) throw new ArgumentOutOfRangeException(nameof(perPage));
        string url = $"https://api.openalex.org/works?search={Uri.EscapeDataString(search)}&per-page={perPage}";
        return ParseWorks(await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses an OpenAlex works response (offline).</summary>
    public static IReadOnlyList<ScholarlyWork> ParseWorks(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);

        var works = new List<ScholarlyWork>();
        if (document.RootElement.TryGetProperty("results", out JsonElement results) && results.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement work in results.EnumerateArray())
            {
                var authors = new List<string>();
                if (work.TryGetProperty("authorships", out JsonElement authorships) && authorships.ValueKind == JsonValueKind.Array)
                {
                    foreach (JsonElement authorship in authorships.EnumerateArray())
                        if (authorship.TryGetProperty("author", out var author) &&
                            author.TryGetProperty("display_name", out var name) && name.ValueKind == JsonValueKind.String)
                            authors.Add(name.GetString()!);
                }

                works.Add(new ScholarlyWork(
                    Id: GetString(work, "id") ?? string.Empty,
                    Title: GetString(work, "title") ?? string.Empty,
                    Year: work.TryGetProperty("publication_year", out var year) && year.ValueKind == JsonValueKind.Number ? year.GetInt32() : null,
                    Doi: GetString(work, "doi"),
                    Authors: authors));
            }
        }
        return works;
    }

    private static string? GetString(JsonElement parent, string name)
        => parent.TryGetProperty(name, out var element) && element.ValueKind == JsonValueKind.String
            ? element.GetString()
            : null;
}
