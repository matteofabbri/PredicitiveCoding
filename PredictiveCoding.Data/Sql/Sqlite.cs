using System.Globalization;
using Microsoft.Data.Sqlite;
using PredictiveCoding.Core;
using PredictiveCoding.Data.Tabular;

namespace PredictiveCoding.Data.Sql;

/// <summary>
/// Runs SQL queries against a SQLite database, returning either generic rows or a
/// numeric <see cref="TabularData"/> matrix.
/// </summary>
public static class Sqlite
{
    /// <summary>Executes a query and returns the rows as column-keyed dictionaries.</summary>
    public static IReadOnlyList<IReadOnlyDictionary<string, object?>> Query(string connectionString, string sql)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(connectionString);
        ArgumentException.ThrowIfNullOrWhiteSpace(sql);

        using var connection = new SqliteConnection(connectionString);
        connection.Open();
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText = sql;
        using SqliteDataReader reader = command.ExecuteReader();

        var rows = new List<IReadOnlyDictionary<string, object?>>();
        while (reader.Read())
        {
            var row = new Dictionary<string, object?>(reader.FieldCount);
            for (int i = 0; i < reader.FieldCount; i++)
                row[reader.GetName(i)] = reader.IsDBNull(i) ? null : reader.GetValue(i);
            rows.Add(row);
        }
        return rows;
    }

    /// <summary>Executes a query whose columns are numeric and returns a <c>[rows, cols]</c> table.</summary>
    public static TabularData QueryMatrix(string connectionString, string sql)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(connectionString);
        ArgumentException.ThrowIfNullOrWhiteSpace(sql);

        using var connection = new SqliteConnection(connectionString);
        connection.Open();
        using SqliteCommand command = connection.CreateCommand();
        command.CommandText = sql;
        using SqliteDataReader reader = command.ExecuteReader();

        int columnCount = reader.FieldCount;
        var columns = new string[columnCount];
        for (int i = 0; i < columnCount; i++)
            columns[i] = reader.GetName(i);

        var rows = new List<float[]>();
        while (reader.Read())
        {
            var row = new float[columnCount];
            for (int i = 0; i < columnCount; i++)
                row[i] = reader.IsDBNull(i) ? float.NaN : Convert.ToSingle(reader.GetValue(i), CultureInfo.InvariantCulture);
            rows.Add(row);
        }

        if (rows.Count == 0)
            throw new InvalidDataException("The query returned no rows.");

        var data = new float[rows.Count * columnCount];
        for (int r = 0; r < rows.Count; r++)
            Array.Copy(rows[r], 0, data, r * columnCount, columnCount);
        return new TabularData(new DenseTensor<float>(data, rows.Count, columnCount), columns);
    }
}
