using PredictiveCoding.Data.HackerNews;

namespace PredicitiveCoding.UnitTests;

public sealed class HackerNewsSourceTests
{
    private const string ItemJson = """
        {
          "id": 8863, "type": "story", "by": "dhouston", "time": 1175714200,
          "title": "My YC app", "url": "http://www.getdropbox.com/u/2/screencast.html",
          "score": 111, "kids": [ 8952, 9224 ]
        }
        """;

    [Fact]
    public void ParseItem_ReadsFields()
    {
        HackerNewsItem item = HackerNewsSource.ParseItem(ItemJson);

        Assert.Equal(8863, item.Id);
        Assert.Equal("story", item.Type);
        Assert.Equal("dhouston", item.By);
        Assert.Equal("My YC app", item.Title);
        Assert.Equal(111, item.Score);
        Assert.Equal([8952L, 9224L], item.Kids);
        Assert.Equal(DateTimeOffset.FromUnixTimeSeconds(1175714200), item.Time);
    }

    [Fact]
    public void ParseIds_ReadsArray()
    {
        IReadOnlyList<long> ids = HackerNewsSource.ParseIds("[1, 2, 3]");
        Assert.Equal([1L, 2L, 3L], ids);
    }

    [Fact]
    public void ParseItem_RejectsEmpty()
        => Assert.Throws<ArgumentException>(() => HackerNewsSource.ParseItem(""));

    [Fact]
    public async Task GetItemAsync_RejectsNegativeId()
    {
        using var source = new HackerNewsSource();
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => source.GetItemAsync(-1));
    }
}
