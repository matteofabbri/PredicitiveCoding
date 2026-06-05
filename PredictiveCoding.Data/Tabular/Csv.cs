namespace PredictiveCoding.Data.Tabular;

/// <summary>
/// Convenience entry point for parsing CSV/TSV into a numeric table. Backed by
/// <see cref="CsvLoader"/>.
/// </summary>
public static class Csv
{
    /// <summary>Loads and parses a CSV/TSV file.</summary>
    public static TabularData Load(string path, CsvLoadOptions? options = null)
        => CsvLoader.Instance.Load(path, options);

    /// <summary>Parses CSV/TSV content held in a string.</summary>
    public static TabularData Parse(string content, CsvLoadOptions? options = null)
        => CsvLoader.Instance.Parse(content, options);
}
