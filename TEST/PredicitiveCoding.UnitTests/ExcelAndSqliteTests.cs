using ClosedXML.Excel;
using Microsoft.Data.Sqlite;
using PredictiveCoding.Data.Sql;
using PredictiveCoding.Data.Tabular;

namespace PredicitiveCoding.UnitTests;

/// <summary>Offline tests for the Excel (ClosedXML) and SQLite sources, using
/// locally created fixtures.</summary>
public sealed class ExcelAndSqliteTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"pcdata_xlsdb_{Guid.NewGuid():N}");

    public ExcelAndSqliteTests() => Directory.CreateDirectory(_tmp);

    public void Dispose()
    {
        SqliteConnection.ClearAllPools();
        if (Directory.Exists(_tmp)) Directory.Delete(_tmp, true);
    }

    // ── Excel ────────────────────────────────────────────────────────────────

    [Fact]
    public void Excel_Load_ReadsNumericSheet()
    {
        string path = Path.Combine(_tmp, "book.xlsx");
        using (var workbook = new XLWorkbook())
        {
            IXLWorksheet worksheet = workbook.AddWorksheet("Sheet1");
            worksheet.Cell(1, 1).Value = "a";
            worksheet.Cell(1, 2).Value = "b";
            worksheet.Cell(2, 1).Value = 1.0;
            worksheet.Cell(2, 2).Value = 2.0;
            worksheet.Cell(3, 1).Value = 3.0;
            worksheet.Cell(3, 2).Value = 4.0;
            workbook.SaveAs(path);
        }

        TabularData table = Excel.Load(path);

        Assert.Equal(2, table.RowCount);
        Assert.Equal(2, table.ColumnCount);
        Assert.Equal(new[] { "a", "b" }, table.Columns);
        Assert.Equal(1.0f, table.Values[0, 0]);
        Assert.Equal(4.0f, table.Values[1, 1]);
    }

    [Fact]
    public void Excel_Missing_Throws()
        => Assert.Throws<FileNotFoundException>(() => Excel.Load(Path.Combine(_tmp, "nope.xlsx")));

    // ── SQLite ───────────────────────────────────────────────────────────────

    private string CreateDatabase()
    {
        string path = Path.Combine(_tmp, "data.db");
        string connectionString = $"Data Source={path};Pooling=False";
        using var connection = new SqliteConnection(connectionString);
        connection.Open();
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText = "CREATE TABLE t(x REAL, y REAL); INSERT INTO t VALUES (1, 2), (3, 4);";
        command.ExecuteNonQuery();
        return connectionString;
    }

    [Fact]
    public void Sqlite_QueryMatrix_ReturnsTensor()
    {
        string connectionString = CreateDatabase();

        TabularData table = Sqlite.QueryMatrix(connectionString, "SELECT x, y FROM t ORDER BY x");

        Assert.Equal(2, table.RowCount);
        Assert.Equal(new[] { "x", "y" }, table.Columns);
        Assert.Equal(1f, table.Values[0, 0]);
        Assert.Equal(4f, table.Values[1, 1]);
    }

    [Fact]
    public void Sqlite_Query_ReturnsRows()
    {
        string connectionString = CreateDatabase();

        IReadOnlyList<IReadOnlyDictionary<string, object?>> rows =
            Sqlite.Query(connectionString, "SELECT x, y FROM t ORDER BY x");

        Assert.Equal(2, rows.Count);
        Assert.Equal(1.0, Convert.ToDouble(rows[0]["x"]));
    }

    [Fact]
    public void Sqlite_QueryMatrix_NoRows_Throws()
    {
        string connectionString = CreateDatabase();
        Assert.Throws<InvalidDataException>(() => Sqlite.QueryMatrix(connectionString, "SELECT x, y FROM t WHERE x > 100"));
    }
}
