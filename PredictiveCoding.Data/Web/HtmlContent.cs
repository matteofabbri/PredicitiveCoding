namespace PredictiveCoding.Data.Web;

/// <summary>
/// The salient content extracted from an HTML document: its title, the visible
/// text (whitespace-collapsed) and the outgoing links.
/// </summary>
public sealed record HtmlContent(string? Title, string Text, IReadOnlyList<string> Links);
