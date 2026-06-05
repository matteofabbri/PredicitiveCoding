using System.Globalization;
using Parquet.Serialization;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Tabular;

/// <summary>
/// Loads numeric columns from Apache Parquet files using Parquet.Net, into the
/// same <see cref="TabularData"/> model as the CSV/Excel loaders.
/// </summary>
public static class ParquetLoader
{
    /// <summary>Loads a Parquet file into a numeric table.</summary>
    public static TabularData Load(string path)
        => LoadAsync(path).GetAwaiter().GetResult();

    /// <summary>Loads a Parquet file into a numeric table.</summary>
    public static async Task<TabularData> LoadAsync(string path, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("Parquet file not found.", path);

        await using var stream = File.OpenRead(path);
        var result = await ParquetSerializer.DeserializeUntypedAsync(stream, cancellationToken: cancellationToken).ConfigureAwait(false);
        IList<Dictionary<string, object>> rows = result.Data;

        if (rows.Count == 0)
            throw new InvalidDataException("Parquet file contained no rows.");

        string[] columns = [.. rows[0].Keys];
        int columnCount = columns.Length;
        if (columnCount == 0)
            throw new InvalidDataException("Parquet file contained no columns.");

        var data = new float[rows.Count * columnCount];
        for (int r = 0; r < rows.Count; r++)
        {
            for (int c = 0; c < columnCount; c++)
            {
                rows[r].TryGetValue(columns[c], out object? value);
                data[r * columnCount + c] = value is null ? float.NaN : Convert.ToSingle(value, CultureInfo.InvariantCulture);
            }
        }

        return new TabularData(new DenseTensor<float>(data, rows.Count, columnCount), columns);
    }
}
