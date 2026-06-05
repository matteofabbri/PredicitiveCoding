using System.Globalization;
using System.Text.Json;
using AngleSharp.Html.Parser;

namespace PredictiveCoding.Data.Mastodon;

/// <summary>A Mastodon status (post), with its HTML content reduced to plain text.</summary>
public sealed record MastodonStatus(string Id, string? Account, string Text, string? Url, DateTimeOffset? CreatedAt);

/// <summary>
/// Data source for the Mastodon fediverse via the public REST API. JSON parsing
/// is static and runs offline; status HTML is stripped with AngleSharp.
/// </summary>
public sealed class MastodonSource : IDisposable
{
    private static readonly HtmlParser Parser = new();

    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public MastodonSource(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Fetches the public timeline of an instance (e.g. <c>mastodon.social</c>).</summary>
    public async Task<IReadOnlyList<MastodonStatus>> GetPublicTimelineAsync(
        string instance, int limit = 20, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(instance);
        string url = $"https://{instance}/api/v1/timelines/public?limit={limit}";
        return ParseStatuses(await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses a Mastodon statuses array (offline).</summary>
    public static IReadOnlyList<MastodonStatus> ParseStatuses(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        if (document.RootElement.ValueKind != JsonValueKind.Array)
            throw new FormatException("Mastodon timeline response is not an array.");

        var statuses = new List<MastodonStatus>();
        foreach (var status in document.RootElement.EnumerateArray())
        {
            string? account = null;
            if (status.TryGetProperty("account", out var acc) && acc.TryGetProperty("username", out var username) &&
                username.ValueKind == JsonValueKind.String)
                account = username.GetString();

            DateTimeOffset? createdAt = null;
            if (status.TryGetProperty("created_at", out var created) && created.ValueKind == JsonValueKind.String &&
                DateTimeOffset.TryParse(created.GetString(), CultureInfo.InvariantCulture, DateTimeStyles.AssumeUniversal, out var dto))
                createdAt = dto;

            statuses.Add(new MastodonStatus(
                Id: GetString(status, "id") ?? string.Empty,
                Account: account,
                Text: StripHtml(GetString(status, "content")),
                Url: GetString(status, "url"),
                CreatedAt: createdAt));
        }
        return statuses;
    }

    private static string StripHtml(string? html)
    {
        if (string.IsNullOrWhiteSpace(html))
            return string.Empty;
        using var document = Parser.ParseDocument(html);
        string text = document.Body?.TextContent ?? string.Empty;
        return string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
    }

    private static string? GetString(JsonElement parent, string name)
        => parent.TryGetProperty(name, out var element) && element.ValueKind == JsonValueKind.String
            ? element.GetString()
            : null;
}
