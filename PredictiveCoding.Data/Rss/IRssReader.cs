namespace PredictiveCoding.Data.Rss;

/// <summary>
/// Reads and parses RSS / Atom feeds into a normalised <see cref="RssFeed"/>.
/// </summary>
public interface IRssReader
{
    /// <summary>Parses feed XML held in a string.</summary>
    RssFeed ReadFromString(string xml);

    /// <summary>Reads and parses a feed from a local file.</summary>
    RssFeed ReadFromFile(string path);

    /// <summary>Downloads and parses a feed from a URL.</summary>
    Task<RssFeed> ReadAsync(string url, CancellationToken cancellationToken = default);
}
