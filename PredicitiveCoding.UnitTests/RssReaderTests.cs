using PredictiveCoding.Data.Rss;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Offline tests for the RSS/Atom reader: parse sample feeds from strings/files
/// and assert on the normalised model. No network is used (parsing is from
/// in-memory XML), so the tests are deterministic.
/// </summary>
public sealed class RssReaderTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"pcdata_rss_{Guid.NewGuid():N}");

    public RssReaderTests() => Directory.CreateDirectory(_tmp);

    public void Dispose()
    {
        if (Directory.Exists(_tmp)) Directory.Delete(_tmp, true);
    }

    private const string Rss2Sample = """
        <?xml version="1.0" encoding="UTF-8"?>
        <rss version="2.0">
          <channel>
            <title>Example Feed</title>
            <link>https://example.com</link>
            <description>An example RSS feed</description>
            <language>en-us</language>
            <item>
              <title>First post</title>
              <link>https://example.com/1</link>
              <description>Summary of the first post</description>
              <author>alice@example.com</author>
              <category>news</category>
              <pubDate>Mon, 01 Jun 2026 12:00:00 GMT</pubDate>
              <guid>https://example.com/1</guid>
            </item>
            <item>
              <title>Second post</title>
              <link>https://example.com/2</link>
              <description>Summary of the second post</description>
              <pubDate>Tue, 02 Jun 2026 09:30:00 GMT</pubDate>
            </item>
          </channel>
        </rss>
        """;

    private const string AtomSample = """
        <?xml version="1.0" encoding="utf-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <title>Atom Example</title>
          <link href="https://example.com/atom"/>
          <subtitle>An example Atom feed</subtitle>
          <entry>
            <title>Atom entry one</title>
            <link href="https://example.com/atom/1"/>
            <id>urn:uuid:00000000-0000-0000-0000-000000000001</id>
            <updated>2026-06-01T12:00:00Z</updated>
            <summary>Atom summary one</summary>
          </entry>
        </feed>
        """;

    // ── Tests ────────────────────────────────────────────────────────────────

    [Fact]
    public void ReadFromString_Rss2_ParsesChannelAndItems()
    {
        RssFeed feed = RssFeeds.ReadFromString(Rss2Sample);

        Assert.Equal("Example Feed", feed.Title);
        Assert.Equal("https://example.com", feed.Link);
        Assert.Equal(2, feed.Items.Count);

        RssItem first = feed.Items[0];
        Assert.Equal("First post", first.Title);
        Assert.Equal("https://example.com/1", first.Link);
        Assert.Equal("Summary of the first post", first.Summary);
        Assert.Contains("news", first.Categories);
        Assert.True(first.Published.HasValue);
        Assert.Equal(2026, first.Published!.Value.Year);
    }

    [Fact]
    public void ReadFromString_Atom_ParsesEntries()
    {
        RssFeed feed = RssFeeds.ReadFromString(AtomSample);

        Assert.Equal("Atom Example", feed.Title);
        Assert.Single(feed.Items);
        Assert.Equal("Atom entry one", feed.Items[0].Title);
        Assert.Equal("https://example.com/atom/1", feed.Items[0].Link);
    }

    [Fact]
    public void Item_Text_CombinesTitleAndBody()
    {
        RssItem item = RssFeeds.ReadFromString(Rss2Sample).Items[0];

        Assert.Contains("First post", item.Text);
        Assert.Contains("Summary of the first post", item.Text);
    }

    [Fact]
    public void ReadFromFile_ParsesSavedFeed()
    {
        string path = Path.Combine(_tmp, "feed.xml");
        File.WriteAllText(path, Rss2Sample);

        RssFeed feed = RssFeeds.ReadFromFile(path);

        Assert.Equal("Example Feed", feed.Title);
        Assert.Equal(2, feed.Items.Count);
    }

    [Fact]
    public void ReadFromString_RejectsNullEmptyAndWhitespace()
    {
        Assert.Throws<ArgumentNullException>(() => RssFeeds.ReadFromString(null!));
        Assert.Throws<ArgumentException>(() => RssFeeds.ReadFromString(""));
        Assert.Throws<ArgumentException>(() => RssFeeds.ReadFromString("   "));
    }

    [Fact]
    public void ReadFromString_MalformedXml_Throws()
        => Assert.ThrowsAny<Exception>(() => RssFeeds.ReadFromString("<rss><not-closed>"));

    [Fact]
    public void ReadFromFile_MissingFile_Throws()
        => Assert.Throws<FileNotFoundException>(() => RssFeeds.ReadFromFile(Path.Combine(_tmp, "missing.xml")));

    [Fact]
    public async Task ReadAsync_RejectsNullAndEmptyBeforeNetwork()
    {
        await Assert.ThrowsAsync<ArgumentNullException>(() => RssFeeds.ReadAsync(null!));
        await Assert.ThrowsAsync<ArgumentException>(() => RssFeeds.ReadAsync(""));
        await Assert.ThrowsAsync<ArgumentException>(() => RssFeeds.ReadAsync("   "));
    }
}
