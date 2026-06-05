namespace PredictiveCoding.Data.Web;

/// <summary>
/// Parses HTML into <see cref="HtmlContent"/> from a string or a URL.
/// </summary>
public interface IHtmlSource
{
    /// <summary>Parses an HTML string (offline).</summary>
    HtmlContent Parse(string html);

    /// <summary>Downloads and parses an HTML page.</summary>
    Task<HtmlContent> ReadAsync(string url, CancellationToken cancellationToken = default);
}
