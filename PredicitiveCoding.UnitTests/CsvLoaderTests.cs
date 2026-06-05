using PredictiveCoding.Data.Tabular;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Offline tests for the CSV/TSV loader: parse delimited text into a numeric
/// table and assert on shape, values, column selection and error handling.
/// </summary>
public sealed class CsvLoaderTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"pcdata_csv_{Guid.NewGuid():N}");

    public CsvLoaderTests() => Directory.CreateDirectory(_tmp);

    public void Dispose()
    {
        if (Directory.Exists(_tmp)) Directory.Delete(_tmp, true);
    }

    [Fact]
    public void Parse_WithHeader_ReadsValuesAndColumns()
    {
        TabularData table = Csv.Parse("a,b,c\n1,2,3\n4,5,6");

        Assert.Equal(2, table.RowCount);
        Assert.Equal(3, table.ColumnCount);
        Assert.Equal(new[] { "a", "b", "c" }, table.Columns);
        Assert.Equal(1f, table.Values[0, 0]);
        Assert.Equal(6f, table.Values[1, 2]);
    }

    [Fact]
    public void Parse_WithoutHeader_TreatsAllRowsAsData()
    {
        TabularData table = Csv.Parse("1,2\n3,4", new CsvLoadOptions { HasHeader = false });

        Assert.Equal(2, table.RowCount);
        Assert.Equal(2, table.ColumnCount);
        Assert.Equal(new[] { "col0", "col1" }, table.Columns);
    }

    [Fact]
    public void Parse_ColumnSubset_SelectsColumns()
    {
        TabularData table = Csv.Parse("a,b,c\n1,2,3\n4,5,6", new CsvLoadOptions { Columns = [0, 2] });

        Assert.Equal(2, table.ColumnCount);
        Assert.Equal(new[] { "a", "c" }, table.Columns);
        Assert.Equal(3f, table.Values[0, 1]); // source column c, first row
    }

    [Fact]
    public void Parse_Tsv_UsesTabDelimiter()
    {
        TabularData table = Csv.Parse("1\t2\t3", new CsvLoadOptions { HasHeader = false, Delimiter = "\t" });
        Assert.Equal(3, table.ColumnCount);
    }

    [Fact]
    public void Parse_NonNumeric_Throws()
        => Assert.Throws<FormatException>(() => Csv.Parse("1,x,3", new CsvLoadOptions { HasHeader = false }));

    [Fact]
    public void Parse_NoDataRows_Throws()
        => Assert.Throws<InvalidDataException>(() => Csv.Parse("a,b,c"));

    [Fact]
    public void Load_FromFile_Works()
    {
        string path = Path.Combine(_tmp, "data.csv");
        File.WriteAllText(path, "x,y\n0.5,-1.5\n2,3");

        TabularData table = Csv.Load(path);

        Assert.Equal(2, table.RowCount);
        Assert.Equal(0.5f, table.Values[0, 0]);
        Assert.Equal(-1.5f, table.Values[0, 1]);
    }
}
