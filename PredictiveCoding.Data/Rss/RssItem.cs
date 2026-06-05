namespace PredictiveCoding.Data.Rss;

/// <summary>
/// A single entry of an RSS or Atom feed, normalised across feed formats.
/// </summary>
public sealed record RssItem(
    string? Id,
    string? Title,
    string? Summary,
    string? Content,
    string? Link,
    string? Author,
    DateTime? Published,
    IReadOnlyList<string> Categories)
{
    /// <summary>
    /// Human-readable text of the item — the title followed by the full content
    /// (or the summary when no full content is present). Convenient as input to a
    /// tokenizer in the LLM layer.
    /// </summary>
    public string Text
    {
        get
        {
            string? body = !string.IsNullOrWhiteSpace(Content) ? Content : Summary;
            return string.Join("\n", new[] { Title, body }.Where(p => !string.IsNullOrWhiteSpace(p)));
        }
    }
}
