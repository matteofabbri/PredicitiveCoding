using PredictiveCoding.Data.Web;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Offline tests for the AngleSharp-backed HTML source: parse markup from a
/// string and validate the network method's argument checks.
/// </summary>
public sealed class HtmlSourceTests
{
    [Fact]
    public void Parse_ExtractsTitleTextAndLinks()
    {
        const string html =
            "<html><head><title>My Page</title></head>" +
            "<body><p>Hello   world</p><a href=\"https://example.com\">link</a></body></html>";

        using var source = new HtmlSource();
        HtmlContent content = source.Parse(html);

        Assert.Equal("My Page", content.Title);
        Assert.Contains("Hello world", content.Text); // whitespace collapsed
        Assert.Contains("https://example.com", content.Links);
    }

    [Fact]
    public void Parse_NoTitle_YieldsNullTitle()
    {
        using var source = new HtmlSource();
        HtmlContent content = source.Parse("<html><body>text only</body></html>");
        Assert.Null(content.Title);
        Assert.Contains("text only", content.Text);
    }

    [Fact]
    public void Parse_Null_Throws()
    {
        using var source = new HtmlSource();
        Assert.Throws<ArgumentNullException>(() => source.Parse(null!));
    }

    [Fact]
    public async Task ReadAsync_RejectsNullAndEmpty()
    {
        using var source = new HtmlSource();
        await Assert.ThrowsAsync<ArgumentNullException>(() => source.ReadAsync(null!));
        await Assert.ThrowsAsync<ArgumentException>(() => source.ReadAsync(""));
        await Assert.ThrowsAsync<ArgumentException>(() => source.ReadAsync("   "));
    }
}
