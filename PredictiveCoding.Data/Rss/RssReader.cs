using CodeHollow.FeedReader;

namespace PredictiveCoding.Data.Rss;

/// <summary>
/// <see cref="IRssReader"/> backed by CodeHollow.FeedReader. Handles RSS
/// 0.91/1.0/2.0 and Atom, tolerating the quirks of real-world feeds.
/// </summary>
public sealed class RssReader : IRssReader
{
    /// <summary>Shared, stateless instance.</summary>
    public static RssReader Instance { get; } = new();

    /// <inheritdoc />
    public RssFeed ReadFromString(string xml)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(xml);
        return Convert(FeedReader.ReadFromString(xml));
    }

    /// <inheritdoc />
    public RssFeed ReadFromFile(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("RSS/Atom feed file not found.", path);
        return Convert(FeedReader.ReadFromFile(path));
    }

    /// <inheritdoc />
    public async Task<RssFeed> ReadAsync(string url, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(url);
        Feed feed = await FeedReader.ReadAsync(url, cancellationToken).ConfigureAwait(false);
        return Convert(feed);
    }

    private static RssFeed Convert(Feed feed)
    {
        var items = new List<RssItem>();
        if (feed.Items is not null)
        {
            foreach (FeedItem item in feed.Items)
            {
                items.Add(new RssItem(
                    Id: item.Id,
                    Title: item.Title,
                    Summary: item.Description,
                    Content: item.Content,
                    Link: item.Link,
                    Author: item.Author,
                    Published: item.PublishingDate,
                    Categories: item.Categories?.ToList() ?? []));
            }
        }

        return new RssFeed(feed.Title, feed.Description, feed.Link, feed.Language, items);
    }
}
