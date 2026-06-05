using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Tabular;

/// <summary>
/// A numeric table: the values as a 2-D <see cref="DenseTensor{T}"/>
/// (<c>[rows, columns]</c>) plus the column names.
/// </summary>
public sealed class TabularData
{
    /// <summary>The cell values, shape <c>[rows, columns]</c>.</summary>
    public DenseTensor<float> Values { get; }

    /// <summary>The column names (header, or generated <c>col0…</c> names).</summary>
    public IReadOnlyList<string> Columns { get; }

    /// <summary>Number of data rows.</summary>
    public int RowCount => Values.Dimensions[0];

    /// <summary>Number of columns.</summary>
    public int ColumnCount => Values.Dimensions[1];

    public TabularData(DenseTensor<float> values, IReadOnlyList<string> columns)
    {
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(columns);
        if (values.Rank != 2)
            throw new ArgumentException("Values must be a 2-D tensor.", nameof(values));
        if (columns.Count != values.Dimensions[1])
            throw new ArgumentException(
                $"Column-name count {columns.Count} does not match the tensor's column count {values.Dimensions[1]}.",
                nameof(columns));

        Values = values;
        Columns = columns;
    }
}
