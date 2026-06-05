using AngleSharp.Html.Parser;

namespace PredictiveCoding.Data.Web;

/// <summary>
/// <see cref="IHtmlSource"/> backed by AngleSharp. Extracts title, visible text
/// and links from HTML — useful as raw text input to the LLM tokenizers.
/// </summary>
public sealed class HtmlSource : IHtmlSource, IDisposable
{
    private const string DefaultUserAgent = "PredictiveCoding.Data/1.0 (HTML data source)";

    private readonly HtmlParser _parser = new();
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;
    private readonly string _userAgent;

    public HtmlSource(HttpClient? httpClient = null, string? userAgent = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
        _userAgent = string.IsNullOrWhiteSpace(userAgent) ? DefaultUserAgent : userAgent;
    }

    /// <inheritdoc />
    public HtmlContent Parse(string html)
    {
        ArgumentNullException.ThrowIfNull(html);
        using var document = _parser.ParseDocument(html);

        string? title = string.IsNullOrWhiteSpace(document.Title) ? null : document.Title;
        string text = CollapseWhitespace(document.Body?.TextContent ?? string.Empty);

        var links = new List<string>();
        foreach (var anchor in document.QuerySelectorAll("a"))
        {
            string? href = anchor.GetAttribute("href");
            if (!string.IsNullOrWhiteSpace(href))
                links.Add(href);
        }

        return new HtmlContent(title, text, links);
    }

    /// <inheritdoc />
    public async Task<HtmlContent> ReadAsync(string url, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(url);

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        request.Headers.UserAgent.ParseAdd(_userAgent);

        using HttpResponseMessage response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        string html = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        return Parse(html);
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    private static string CollapseWhitespace(string text)
        => string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries));
}
