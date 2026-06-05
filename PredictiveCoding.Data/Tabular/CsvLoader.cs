using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Tabular;

/// <summary>
/// <see cref="ICsvLoader"/> backed by CsvHelper. Parses delimited text into a
/// numeric matrix; every selected cell must be parseable as a float.
/// </summary>
public sealed class CsvLoader : ICsvLoader
{
    /// <summary>Shared, stateless instance.</summary>
    public static CsvLoader Instance { get; } = new();

    /// <inheritdoc />
    public TabularData Load(string path, CsvLoadOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("CSV file not found.", path);
        using var reader = new StreamReader(path);
        return Read(reader, options ?? CsvLoadOptions.Default);
    }

    /// <inheritdoc />
    public TabularData Parse(string content, CsvLoadOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(content);
        using var reader = new StringReader(content);
        return Read(reader, options ?? CsvLoadOptions.Default);
    }

    private static TabularData Read(TextReader textReader, CsvLoadOptions options)
    {
        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            Delimiter = options.Delimiter,
            HasHeaderRecord = options.HasHeader,
            DetectColumnCountChanges = false,
            MissingFieldFound = null,
            BadDataFound = null,
        };

        string[]? header = null;
        var rows = new List<string[]>();
        using (var parser = new CsvParser(textReader, config))
        {
            while (parser.Read())
            {
                string[] record = parser.Record ?? [];
                if (options.HasHeader && header is null)
                {
                    header = record;
                    continue;
                }
                rows.Add(record);
            }
        }

        if (rows.Count == 0)
            throw new InvalidDataException("CSV contains no data rows.");

        int sourceColumns = rows[0].Length;
        int[] indices = options.Columns ?? [.. Enumerable.Range(0, sourceColumns)];
        foreach (int index in indices)
            if (index < 0 || index >= sourceColumns)
                throw new ArgumentOutOfRangeException(nameof(options),
                    $"Column index {index} is out of range (0..{sourceColumns - 1}).");

        int rowCount = rows.Count;
        int columnCount = indices.Length;
        var data = new float[rowCount * columnCount];

        for (int r = 0; r < rowCount; r++)
        {
            string[] row = rows[r];
            for (int c = 0; c < columnCount; c++)
            {
                int sourceIndex = indices[c];
                string cell = sourceIndex < row.Length ? row[sourceIndex] : string.Empty;
                if (!float.TryParse(cell, NumberStyles.Float | NumberStyles.AllowThousands,
                        CultureInfo.InvariantCulture, out float value))
                    throw new FormatException($"Non-numeric value '{cell}' at row {r}, column {sourceIndex}.");
                data[r * columnCount + c] = value;
            }
        }

        var values = new DenseTensor<float>(data, rowCount, columnCount);
        string[] columns = [.. indices.Select(i => header is not null && i < header.Length ? header[i] : $"col{i}")];
        return new TabularData(values, columns);
    }
}
