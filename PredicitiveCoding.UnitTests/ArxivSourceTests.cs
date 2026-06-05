using PredictiveCoding.Data.ArXiv;

namespace PredicitiveCoding.UnitTests;

public sealed class ArxivSourceTests
{
    private const string AtomSample = """
        <?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
          <entry>
            <id>http://arxiv.org/abs/1999.12345v1</id>
            <title>Predictive Coding in the Cortex</title>
            <summary>We study predictive coding.</summary>
            <published>1999-01-15T00:00:00Z</published>
            <author><name>Rajesh Rao</name></author>
            <author><name>Dana Ballard</name></author>
            <link href="http://arxiv.org/abs/1999.12345v1" rel="alternate" type="text/html"/>
            <link title="pdf" href="http://arxiv.org/pdf/1999.12345v1" rel="related" type="application/pdf"/>
            <category term="q-bio.NC" scheme="http://arxiv.org/schemas/atom"/>
          </entry>
        </feed>
        """;

    [Fact]
    public void ParseFeed_ReadsEntries()
    {
        IReadOnlyList<ArxivEntry> entries = ArxivSource.ParseFeed(AtomSample);

        ArxivEntry entry = Assert.Single(entries);
        Assert.Equal("http://arxiv.org/abs/1999.12345v1", entry.Id);
        Assert.Equal("Predictive Coding in the Cortex", entry.Title);
        Assert.Contains("predictive coding", entry.Summary);
        Assert.Equal(["Rajesh Rao", "Dana Ballard"], entry.Authors);
        Assert.Equal("http://arxiv.org/pdf/1999.12345v1", entry.PdfUrl);
        Assert.Contains("q-bio.NC", entry.Categories);
        Assert.Equal(1999, entry.Published!.Value.Year);
    }

    [Fact]
    public void ParseFeed_RejectsEmpty()
        => Assert.Throws<ArgumentException>(() => ArxivSource.ParseFeed(""));

    [Fact]
    public async Task SearchAsync_ValidatesArguments()
    {
        using var source = new ArxivSource();
        await Assert.ThrowsAsync<ArgumentException>(() => source.SearchAsync(""));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => source.SearchAsync("x", 0));
    }
}
