using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;

namespace PredictiveCoding.Data.Datasets;

/// <summary>
/// A delimited dataset parsed as <b>strings</b>: a header plus the data rows.
/// Unlike <see cref="PredictiveCoding.Data.Tabular.TabularData"/> (which is
/// numeric-only) this keeps every cell as text, because the canonical R and
/// Python teaching datasets mix numeric, factor and string columns (species
/// names, country codes, …). Numeric columns can be projected into a
/// <c>DenseTensor</c> afterwards via <see cref="PredictiveCoding.Data.Tabular.Csv"/>.
/// </summary>
public sealed class DatasetTable
{
    /// <summary>The column names (the first CSV record).</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <summary>The data rows, each aligned to <see cref="Columns"/>.</summary>
    public IReadOnlyList<IReadOnlyList<string>> Rows { get; }

    /// <summary>Number of data rows.</summary>
    public int RowCount => Rows.Count;

    /// <summary>Number of columns.</summary>
    public int ColumnCount => Columns.Count;

    public DatasetTable(IReadOnlyList<string> columns, IReadOnlyList<IReadOnlyList<string>> rows)
    {
        ArgumentNullException.ThrowIfNull(columns);
        ArgumentNullException.ThrowIfNull(rows);
        Columns = columns;
        Rows = rows;
    }

    /// <summary>Returns the index of a column by name (case-insensitive), or -1 if absent.</summary>
    public int IndexOf(string column)
    {
        ArgumentNullException.ThrowIfNull(column);
        for (int i = 0; i < Columns.Count; i++)
            if (string.Equals(Columns[i], column, StringComparison.OrdinalIgnoreCase))
                return i;
        return -1;
    }

    /// <summary>Returns every value of a named column (empty string where a row is short).</summary>
    public IReadOnlyList<string> Column(string name)
    {
        int index = IndexOf(name);
        if (index < 0)
            throw new ArgumentException($"Column '{name}' is not present.", nameof(name));
        var values = new string[Rows.Count];
        for (int r = 0; r < Rows.Count; r++)
            values[r] = index < Rows[r].Count ? Rows[r][index] : string.Empty;
        return values;
    }
}

/// <summary>Parses delimited dataset text into a <see cref="DatasetTable"/> (offline,
/// string-typed). Backed by CsvHelper, so quoted fields and embedded delimiters
/// are handled correctly.</summary>
public static class DatasetCsv
{
    /// <summary>Parses CSV/TSV content; the first record is treated as the header.</summary>
    public static DatasetTable Parse(string content, string delimiter = ",")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(content);

        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            Delimiter = delimiter,
            HasHeaderRecord = false, // the header is taken as the first record below
            DetectColumnCountChanges = false,
            MissingFieldFound = null,
            BadDataFound = null,
        };

        using var reader = new StringReader(content);
        using var parser = new CsvParser(reader, config);

        string[]? header = null;
        var rows = new List<IReadOnlyList<string>>();
        while (parser.Read())
        {
            string[] record = parser.Record ?? [];
            if (header is null)
            {
                header = record;
                continue;
            }
            rows.Add(record);
        }

        if (header is null)
            throw new InvalidDataException("CSV contains no header row.");
        return new DatasetTable(header, rows);
    }
}
