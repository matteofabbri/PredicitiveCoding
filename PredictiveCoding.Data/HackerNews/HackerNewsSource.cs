using System.Text.Json;

namespace PredictiveCoding.Data.HackerNews;

/// <summary>An item (story, comment, …) from the Hacker News API.</summary>
public sealed record HackerNewsItem(
    long Id,
    string? Type,
    string? By,
    DateTimeOffset? Time,
    string? Title,
    string? Text,
    string? Url,
    int? Score,
    IReadOnlyList<long> Kids);

/// <summary>
/// Data source for Hacker News via its public Firebase API. JSON parsing is
/// static and runs offline.
/// </summary>
public sealed class HackerNewsSource : IDisposable
{
    private const string BaseUrl = "https://hacker-news.firebaseio.com/v0/";

    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public HackerNewsSource(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Fetches a single item by id.</summary>
    public async Task<HackerNewsItem> GetItemAsync(long id, CancellationToken cancellationToken = default)
    {
        if (id < 0) throw new ArgumentOutOfRangeException(nameof(id));
        string json = await _http.GetStringAsync($"{BaseUrl}item/{id}.json", cancellationToken).ConfigureAwait(false);
        return ParseItem(json);
    }

    /// <summary>Fetches the current top-story ids.</summary>
    public async Task<IReadOnlyList<long>> GetTopStoryIdsAsync(CancellationToken cancellationToken = default)
    {
        string json = await _http.GetStringAsync($"{BaseUrl}topstories.json", cancellationToken).ConfigureAwait(false);
        return ParseIds(json);
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses an item JSON document (offline).</summary>
    public static HackerNewsItem ParseItem(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        long id = root.TryGetProperty("id", out var idEl) ? idEl.GetInt64() : 0;
        DateTimeOffset? time = root.TryGetProperty("time", out var t) && t.ValueKind == JsonValueKind.Number
            ? DateTimeOffset.FromUnixTimeSeconds(t.GetInt64())
            : null;

        var kids = new List<long>();
        if (root.TryGetProperty("kids", out var kidsEl) && kidsEl.ValueKind == JsonValueKind.Array)
            foreach (var k in kidsEl.EnumerateArray())
                kids.Add(k.GetInt64());

        return new HackerNewsItem(
            id,
            GetString(root, "type"),
            GetString(root, "by"),
            time,
            GetString(root, "title"),
            GetString(root, "text"),
            GetString(root, "url"),
            root.TryGetProperty("score", out var s) && s.ValueKind == JsonValueKind.Number ? s.GetInt32() : null,
            kids);
    }

    /// <summary>Parses an array of ids (offline).</summary>
    public static IReadOnlyList<long> ParseIds(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        var ids = new List<long>();
        if (document.RootElement.ValueKind == JsonValueKind.Array)
            foreach (var element in document.RootElement.EnumerateArray())
                ids.Add(element.GetInt64());
        return ids;
    }

    private static string? GetString(JsonElement parent, string name)
        => parent.TryGetProperty(name, out var element) && element.ValueKind == JsonValueKind.String
            ? element.GetString()
            : null;
}
