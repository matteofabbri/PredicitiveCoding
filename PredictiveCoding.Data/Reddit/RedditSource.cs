using System.Text.Json;

namespace PredictiveCoding.Data.Reddit;

/// <summary>A Reddit post (link or self-post).</summary>
public sealed record RedditPost(string Id, string? Title, string? Author, string? Url, int Score, string? Subreddit, string? SelfText);

/// <summary>
/// Data source for Reddit via its public JSON endpoints. JSON parsing is static
/// and runs offline. A descriptive User-Agent is required by Reddit.
/// </summary>
public sealed class RedditSource : IDisposable
{
    private const string DefaultUserAgent = "PredictiveCoding.Data/1.0 (Reddit data source)";

    private readonly HttpClient _http;
    private readonly bool _ownsHttp;
    private readonly string _userAgent;

    public RedditSource(HttpClient? httpClient = null, string? userAgent = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
        _userAgent = string.IsNullOrWhiteSpace(userAgent) ? DefaultUserAgent : userAgent;
    }

    /// <summary>Fetches posts of a subreddit (<paramref name="sort"/>: hot/new/top/…).</summary>
    public async Task<IReadOnlyList<RedditPost>> GetPostsAsync(
        string subreddit, string sort = "hot", int limit = 25, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(subreddit);
        ArgumentException.ThrowIfNullOrWhiteSpace(sort);
        string url = $"https://www.reddit.com/r/{Uri.EscapeDataString(subreddit)}/{Uri.EscapeDataString(sort)}.json?limit={limit}";

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        request.Headers.UserAgent.ParseAdd(_userAgent);
        using HttpResponseMessage response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return ParsePosts(await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false));
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses a Reddit listing response (offline).</summary>
    public static IReadOnlyList<RedditPost> ParsePosts(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);

        var posts = new List<RedditPost>();
        if (document.RootElement.TryGetProperty("data", out var data) &&
            data.TryGetProperty("children", out var children) && children.ValueKind == JsonValueKind.Array)
        {
            foreach (var child in children.EnumerateArray())
            {
                if (!child.TryGetProperty("data", out var post))
                    continue;
                posts.Add(new RedditPost(
                    Id: GetString(post, "id") ?? string.Empty,
                    Title: GetString(post, "title"),
                    Author: GetString(post, "author"),
                    Url: GetString(post, "url"),
                    Score: post.TryGetProperty("score", out var score) && score.ValueKind == JsonValueKind.Number ? score.GetInt32() : 0,
                    Subreddit: GetString(post, "subreddit"),
                    SelfText: GetString(post, "selftext")));
            }
        }
        return posts;
    }

    private static string? GetString(JsonElement parent, string name)
        => parent.TryGetProperty(name, out var element) && element.ValueKind == JsonValueKind.String
            ? element.GetString()
            : null;
}
