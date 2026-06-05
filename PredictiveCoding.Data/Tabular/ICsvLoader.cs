namespace PredictiveCoding.Data.Tabular;

/// <summary>
/// Parses CSV/TSV data into a numeric <see cref="TabularData"/> table.
/// </summary>
public interface ICsvLoader
{
    /// <summary>Loads and parses a CSV/TSV file.</summary>
    TabularData Load(string path, CsvLoadOptions? options = null);

    /// <summary>Parses CSV/TSV content held in a string.</summary>
    TabularData Parse(string content, CsvLoadOptions? options = null);
}
