using PredictiveCoding.Data.Wikipedia;

namespace PredicitiveCoding.UnitTests;

public sealed class WikipediaSourceTests
{
    [Fact]
    public void ParseSummary_ReadsFields()
    {
        const string json = """
            {
              "title": "Predictive coding",
              "description": "theory of brain function",
              "extract": "Predictive coding is a theory of brain function.",
              "content_urls": { "desktop": { "page": "https://en.wikipedia.org/wiki/Predictive_coding" } }
            }
            """;

        WikipediaPage page = WikipediaSource.ParseSummary(json);

        Assert.Equal("Predictive coding", page.Title);
        Assert.Equal("theory of brain function", page.Description);
        Assert.Contains("brain function", page.Extract);
        Assert.Equal("https://en.wikipedia.org/wiki/Predictive_coding", page.Url);
    }

    [Fact]
    public void ParseExtract_ReadsFirstPageExtract()
    {
        const string json = """
            { "query": { "pages": { "12345": { "pageid": 12345, "title": "X", "extract": "Full plain text." } } } }
            """;
        Assert.Equal("Full plain text.", WikipediaSource.ParseExtract(json));
    }

    [Fact]
    public void ParseSearch_ReadsTitles()
    {
        const string json = """
            { "query": { "search": [ { "title": "Alpha" }, { "title": "Beta" } ] } }
            """;
        Assert.Equal(["Alpha", "Beta"], WikipediaSource.ParseSearch(json));
    }

    [Fact]
    public async Task Methods_ValidateArguments()
    {
        using var source = new WikipediaSource();
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetSummaryAsync(""));
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetExtractAsync("  "));
        await Assert.ThrowsAsync<ArgumentException>(() => source.SearchAsync(""));
    }
}
