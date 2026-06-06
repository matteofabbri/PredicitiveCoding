using PredictiveCoding.Data.Text;

namespace PredicitiveCoding.UnitTests;

public sealed class TextCorpusTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"pcdata_text_{Guid.NewGuid():N}");

    public TextCorpusTests() => Directory.CreateDirectory(_tmp);

    public void Dispose()
    {
        if (Directory.Exists(_tmp)) Directory.Delete(_tmp, true);
    }

    [Fact]
    public void ReadFile_ReturnsContent()
    {
        string path = Path.Combine(_tmp, "a.txt");
        File.WriteAllText(path, "hello");
        TextDocument document = TextCorpus.ReadFile(path);
        Assert.Equal("hello", document.Text);
        Assert.Equal(path, document.Path);
    }

    [Fact]
    public void ReadDirectory_ReadsMatchingFilesSorted()
    {
        File.WriteAllText(Path.Combine(_tmp, "b.txt"), "B");
        File.WriteAllText(Path.Combine(_tmp, "a.txt"), "A");
        File.WriteAllText(Path.Combine(_tmp, "skip.md"), "M");

        IReadOnlyList<TextDocument> docs = TextCorpus.ReadDirectory(_tmp, "*.txt");

        Assert.Equal(2, docs.Count);
        Assert.EndsWith("a.txt", docs[0].Path);
        Assert.Equal("A", docs[0].Text);
    }

    [Fact]
    public void ReadFile_Missing_Throws()
        => Assert.Throws<FileNotFoundException>(() => TextCorpus.ReadFile(Path.Combine(_tmp, "nope.txt")));

    [Fact]
    public void ReadDirectory_Missing_Throws()
        => Assert.Throws<DirectoryNotFoundException>(() => TextCorpus.ReadDirectory(Path.Combine(_tmp, "nodir")));
}
