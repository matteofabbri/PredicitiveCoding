namespace PredictiveCoding.Data.Tabular;

/// <summary>
/// Options controlling how a CSV/TSV file is parsed into a numeric table.
/// </summary>
public sealed class CsvLoadOptions
{
    /// <summary>Whether the first row is a header (default true).</summary>
    public bool HasHeader { get; init; } = true;

    /// <summary>Field delimiter (default <c>","</c>; use <c>"\t"</c> for TSV).</summary>
    public string Delimiter { get; init; } = ",";

    /// <summary>
    /// Indices of the source columns to keep (in order). <c>null</c> keeps every column.
    /// </summary>
    public int[]? Columns { get; init; }

    /// <summary>Shared instance with the default settings.</summary>
    public static CsvLoadOptions Default { get; } = new();
}
