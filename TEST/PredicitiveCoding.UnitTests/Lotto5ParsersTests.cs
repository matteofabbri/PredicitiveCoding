using PredictiveCoding.Data.Blob;
using PredictiveCoding.Data.ConceptNet;
using PredictiveCoding.Data.HuggingFace;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Offline tests for the final batch-5 sources: ConceptNet and Hugging Face JSON
/// parsers, and Blob downloader argument validation.
/// </summary>
public sealed class Lotto5ParsersTests
{
    [Fact]
    public void ConceptNet_ParseEdges()
    {
        const string json = """
            { "edges": [ { "start": { "label": "dog" }, "end": { "label": "animal" },
              "rel": { "label": "IsA" }, "weight": 2.5 } ] }
            """;
        ConceptNetEdge edge = Assert.Single(ConceptNetSource.ParseEdges(json));
        Assert.Equal("dog", edge.Start);
        Assert.Equal("animal", edge.End);
        Assert.Equal("IsA", edge.Relation);
        Assert.Equal(2.5, edge.Weight);
    }

    [Fact]
    public void ConceptNet_ParseEdges_RejectsEmpty()
        => Assert.Throws<ArgumentException>(() => ConceptNetSource.ParseEdges(""));

    [Fact]
    public void HuggingFace_ParseRows()
    {
        const string json = """
            { "rows": [
                { "row_idx": 0, "row": { "text": "hello", "label": 1 } },
                { "row_idx": 1, "row": { "text": "world", "label": 0 } } ] }
            """;
        IReadOnlyList<IReadOnlyDictionary<string, object?>> rows = HuggingFaceDatasets.ParseRows(json);

        Assert.Equal(2, rows.Count);
        Assert.Equal("hello", rows[0]["text"]);
        Assert.Equal(1L, rows[0]["label"]);
        Assert.Equal("world", rows[1]["text"]);
    }

    [Fact]
    public async Task HuggingFace_GetRows_ValidatesArguments()
    {
        using var source = new HuggingFaceDatasets();
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetRowsAsync(""));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => source.GetRowsAsync("ds", length: 0));
    }

    [Fact]
    public async Task Blob_Download_ValidatesArguments()
    {
        using var downloader = new BlobDownloader();
        await Assert.ThrowsAsync<ArgumentException>(() => downloader.DownloadBytesAsync(""));
        await Assert.ThrowsAsync<ArgumentException>(() => downloader.DownloadToFileAsync("", "out.bin"));
        await Assert.ThrowsAsync<ArgumentException>(() => downloader.DownloadToFileAsync("https://example.com/x", ""));
    }
}
