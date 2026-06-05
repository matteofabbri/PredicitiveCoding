namespace PredictiveCoding.Data.Rss;

/// <summary>
/// A parsed RSS or Atom feed: channel-level metadata plus its items.
/// </summary>
public sealed record RssFeed(
    string? Title,
    string? Description,
    string? Link,
    string? Language,
    IReadOnlyList<RssItem> Items);
