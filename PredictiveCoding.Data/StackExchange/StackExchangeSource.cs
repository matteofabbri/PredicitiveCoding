using System.Net;
using System.Text.Json;

namespace PredictiveCoding.Data.StackExchange;

/// <summary>A question from the Stack Exchange API.</summary>
public sealed record StackExchangeQuestion(long Id, string? Title, string? Link, int Score, IReadOnlyList<string> Tags);

/// <summary>
/// Data source for the Stack Exchange network (Stack Overflow, etc.). JSON
/// parsing is static and runs offline; the API always gzips responses, so the
/// default client enables automatic decompression.
/// </summary>
public sealed class StackExchangeSource : IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public StackExchangeSource(HttpClient? httpClient = null)
    {
        if (httpClient is null)
        {
            _http = new HttpClient(new HttpClientHandler
            {
                AutomaticDecompression = DecompressionMethods.GZip | DecompressionMethods.Deflate,
            });
            _ownsHttp = true;
        }
        else
        {
            _http = httpClient;
            _ownsHttp = false;
        }
    }

    /// <summary>Searches questions on a site (default <c>stackoverflow</c>).</summary>
    public async Task<IReadOnlyList<StackExchangeQuestion>> SearchAsync(
        string query, string site = "stackoverflow", CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        ArgumentException.ThrowIfNullOrWhiteSpace(site);
        string url = $"https://api.stackexchange.com/2.3/search/advanced?order=desc&sort=relevance"
                   + $"&site={Uri.EscapeDataString(site)}&q={Uri.EscapeDataString(query)}";
        return ParseQuestions(await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses a Stack Exchange search response (offline).</summary>
    public static IReadOnlyList<StackExchangeQuestion> ParseQuestions(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);

        var questions = new List<StackExchangeQuestion>();
        if (document.RootElement.TryGetProperty("items", out JsonElement items) && items.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement item in items.EnumerateArray())
            {
                var tags = new List<string>();
                if (item.TryGetProperty("tags", out JsonElement tagArray) && tagArray.ValueKind == JsonValueKind.Array)
                    foreach (JsonElement tag in tagArray.EnumerateArray())
                        if (tag.ValueKind == JsonValueKind.String)
                            tags.Add(tag.GetString()!);

                questions.Add(new StackExchangeQuestion(
                    Id: item.TryGetProperty("question_id", out var id) && id.ValueKind == JsonValueKind.Number ? id.GetInt64() : 0,
                    Title: GetString(item, "title"),
                    Link: GetString(item, "link"),
                    Score: item.TryGetProperty("score", out var score) && score.ValueKind == JsonValueKind.Number ? score.GetInt32() : 0,
                    Tags: tags));
            }
        }
        return questions;
    }

    private static string? GetString(JsonElement parent, string name)
        => parent.TryGetProperty(name, out var element) && element.ValueKind == JsonValueKind.String
            ? element.GetString()
            : null;
}
