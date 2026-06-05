namespace PredictiveCoding.Data.Rss;

/// <summary>
/// Convenience entry point for reading RSS / Atom feeds. Backed by
/// <see cref="RssReader"/>.
/// </summary>
public static class RssFeeds
{
    /// <summary>Parses feed XML held in a string.</summary>
    public static RssFeed ReadFromString(string xml)
        => RssReader.Instance.ReadFromString(xml);

    /// <summary>Reads and parses a feed from a local file.</summary>
    public static RssFeed ReadFromFile(string path)
        => RssReader.Instance.ReadFromFile(path);

    /// <summary>Downloads and parses a feed from a URL.</summary>
    public static Task<RssFeed> ReadAsync(string url, CancellationToken cancellationToken = default)
        => RssReader.Instance.ReadAsync(url, cancellationToken);
}
