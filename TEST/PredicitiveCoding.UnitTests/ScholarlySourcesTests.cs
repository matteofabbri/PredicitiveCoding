using PredictiveCoding.Data.Scholarly;

namespace PredicitiveCoding.UnitTests;

/// <summary>Offline tests for the ScienceDirect (Elsevier API) and Unpaywall
/// scholarly sources: JSON parsing and argument validation. No network is used.</summary>
public sealed class ScholarlySourcesTests
{
    // ── ScienceDirect ────────────────────────────────────────────────────────

    [Fact]
    public void ScienceDirect_ParseSearch()
    {
        const string json = """
            { "search-results": { "entry": [ {
                "dc:title": "Predictive coding networks",
                "dc:creator": "Rao R.",
                "prism:doi": "10.1016/j.x.2020.01",
                "prism:publicationName": "Neuron",
                "prism:coverDate": "2020-01-15",
                "prism:url": "https://api.elsevier.com/content/article/pii/S123" } ] } }
            """;
        ScienceDirectArticle article = Assert.Single(ScienceDirectSource.ParseSearch(json));
        Assert.Equal("Predictive coding networks", article.Title);
        Assert.Equal("Rao R.", article.Authors);
        Assert.Equal("10.1016/j.x.2020.01", article.Doi);
        Assert.Equal("Neuron", article.PublicationName);
    }

    [Fact]
    public void ScienceDirect_ParseArticle_WithAuthorArray()
    {
        const string json = """
            { "full-text-retrieval-response": { "coredata": {
                "dc:title": "A study",
                "prism:doi": "10.1016/x",
                "dc:creator": [ { "$": "Smith J." }, { "$": "Doe A." } ],
                "prism:publicationName": "Journal X",
                "prism:coverDate": "2021-03-01" } } }
            """;
        ScienceDirectArticle article = ScienceDirectSource.ParseArticle(json);
        Assert.Equal("A study", article.Title);
        Assert.Equal("10.1016/x", article.Doi);
        Assert.Equal("Smith J., Doe A.", article.Authors);
    }

    [Fact]
    public async Task ScienceDirect_ConstructionAndValidation()
    {
        Assert.Throws<ArgumentException>(() => new ScienceDirectSource(""));
        using var source = new ScienceDirectSource("dummy-key");
        await Assert.ThrowsAsync<ArgumentException>(() => source.SearchAsync(""));
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetByDoiAsync(""));
        Assert.Throws<ArgumentException>(() => ScienceDirectSource.ParseSearch(""));
    }

    // ── Unpaywall ────────────────────────────────────────────────────────────

    [Fact]
    public void Unpaywall_Parse()
    {
        const string json = """
            { "doi": "10.1016/x", "title": "A paper", "is_oa": true,
              "best_oa_location": { "url_for_pdf": "https://repo/x.pdf", "url": "https://repo/x" },
              "oa_locations": [ { "url_for_pdf": "https://repo/x.pdf" }, { "url_for_pdf": null, "url": "https://repo2/x" } ] }
            """;
        UnpaywallRecord record = UnpaywallSource.Parse(json);
        Assert.Equal("10.1016/x", record.Doi);
        Assert.True(record.IsOpenAccess);
        Assert.Equal("https://repo/x.pdf", record.BestPdfUrl);
        Assert.Equal(["https://repo/x.pdf", "https://repo2/x"], record.PdfUrls);
    }

    [Fact]
    public async Task Unpaywall_ConstructionAndValidation()
    {
        Assert.Throws<ArgumentException>(() => new UnpaywallSource(""));
        using var source = new UnpaywallSource("you@example.com");
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetAsync(""));
        Assert.Throws<ArgumentException>(() => UnpaywallSource.Parse(""));
    }
}
