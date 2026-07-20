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

    /// <summary>
    /// Block-aware extraction (ROADMAP-Scale C1/C2): strips the chrome
    /// (script/style/nav/header/footer/aside/form) from the DOM, then returns
    /// the text of each content block (p, headings, li, blockquote, pre, …) as
    /// its own line. Short blocks are the classic boilerplate signature — the
    /// crawler filters them by word count, and repeated blocks across pages
    /// (menus, cookie banners) fall to the corpus-side line dedup.
    /// </summary>
    public IReadOnlyList<string> ParseBlocks(string html)
    {
        ArgumentNullException.ThrowIfNull(html);
        using var document = _parser.ParseDocument(html);

        foreach (var chrome in document.QuerySelectorAll("script, style, nav, header, footer, aside, form, noscript").ToArray())
            chrome.Remove();

        var blocks = new List<string>();
        foreach (var node in document.QuerySelectorAll("p, h1, h2, h3, h4, h5, h6, li, blockquote, pre, figcaption, dt, dd"))
        {
            string text = CollapseWhitespace(node.TextContent);
            if (text.Length > 0) blocks.Add(text);
        }
        return blocks;
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
