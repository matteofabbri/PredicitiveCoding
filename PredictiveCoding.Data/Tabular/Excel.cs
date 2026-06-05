using System.Globalization;
using ClosedXML.Excel;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Tabular;

/// <summary>
/// Loads numeric data from Excel (.xlsx) worksheets using ClosedXML, into the
/// same <see cref="TabularData"/> model as the CSV loader.
/// </summary>
public static class Excel
{
    /// <summary>Loads a worksheet's used range as a numeric table.</summary>
    public static TabularData Load(string path, string? sheet = null, bool hasHeader = true)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("Excel file not found.", path);

        using var workbook = new XLWorkbook(path);
        IXLWorksheet worksheet = sheet is null ? workbook.Worksheet(1) : workbook.Worksheet(sheet);
        IXLRange? range = worksheet.RangeUsed();
        if (range is null)
            throw new InvalidDataException("The worksheet is empty.");

        int rowCount = range.RowCount();
        int columnCount = range.ColumnCount();
        int firstDataRow = hasHeader ? 2 : 1;
        if (rowCount < firstDataRow)
            throw new InvalidDataException("The worksheet has no data rows.");

        var columns = new string[columnCount];
        for (int c = 1; c <= columnCount; c++)
            columns[c - 1] = hasHeader ? range.Cell(1, c).GetString() : $"col{c - 1}";

        int dataRows = rowCount - (hasHeader ? 1 : 0);
        var data = new float[dataRows * columnCount];
        for (int r = 0; r < dataRows; r++)
        {
            for (int c = 0; c < columnCount; c++)
            {
                string text = range.Cell(firstDataRow + r, c + 1).GetString();
                if (!float.TryParse(text, NumberStyles.Float | NumberStyles.AllowThousands, CultureInfo.InvariantCulture, out float value))
                    throw new FormatException($"Non-numeric cell at row {r}, column {c}: '{text}'.");
                data[r * columnCount + c] = value;
            }
        }

        return new TabularData(new DenseTensor<float>(data, dataRows, columnCount), columns);
    }
}
