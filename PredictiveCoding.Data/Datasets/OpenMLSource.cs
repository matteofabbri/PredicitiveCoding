using System.Globalization;
using System.Text.Json;

namespace PredictiveCoding.Data.Datasets;

/// <summary>One entry of the OpenML data catalogue.</summary>
/// <param name="Id">OpenML data id (<c>did</c>), used to download the dataset.</param>
/// <param name="Name">Dataset name (e.g. <c>iris</c>).</param>
/// <param name="Status">Catalogue status (<c>active</c>, <c>deactivated</c>, …).</param>
public sealed record OpenMLDatasetInfo(int Id, string Name, string Status);

/// <summary>
/// The <b>OpenML</b> source, backed by the OpenML REST API
/// (<c>https://www.openml.org/api/v1</c>) — a large, versioned repository of machine-
/// learning benchmark datasets addressed by a numeric data id (<c>iris</c> = 61,
/// <c>credit-g</c> = 31, <c>adult</c> = 1590, <c>mnist_784</c> = 554, …). Unlike the
/// GitHub-backed catalogues this one has a JSON index and downloads by id; the CSV is
/// parsed into a <see cref="DatasetTable"/>. JSON parsing is static and runs offline.
/// </summary>
public sealed class OpenMLSource : IDisposable
{
    /// <summary>Base URL of the OpenML v1 REST API.</summary>
    public const string ApiBase = "https://www.openml.org/api/v1/json";

    /// <summary>Base URL serving datasets as CSV by id.</summary>
    public const string CsvBase = "https://www.openml.org/data/get_csv";

    /// <summary>A handful of classic datasets with their stable OpenML ids (offline fallback).</summary>
    public static IReadOnlyList<OpenMLDatasetInfo> WellKnownDatasets { get; } =
    [
        new(61, "iris", "active"),
        new(31, "credit-g", "active"),
        new(37, "diabetes", "active"),
        new(1590, "adult", "active"),
        new(554, "mnist_784", "active"),
        new(40975, "car", "active"),
    ];

    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public OpenMLSource(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Lists active datasets from the catalogue (up to <paramref name="limit"/>).</summary>
    public async Task<IReadOnlyList<OpenMLDatasetInfo>> ListDatasetsAsync(int limit = 100, CancellationToken cancellationToken = default)
    {
        if (limit < 1) throw new ArgumentOutOfRangeException(nameof(limit));
        string url = $"{ApiBase}/data/list/limit/{limit.ToString(CultureInfo.InvariantCulture)}";
        return ParseIndex(await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Downloads the raw CSV of a dataset by its OpenML id.</summary>
    public async Task<string> GetCsvAsync(int dataId, CancellationToken cancellationToken = default)
    {
        if (dataId < 1) throw new ArgumentOutOfRangeException(nameof(dataId));
        string url = $"{CsvBase}/{dataId.ToString(CultureInfo.InvariantCulture)}";
        return await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Downloads one dataset by id and parses it into a string-typed table.</summary>
    public async Task<DatasetTable> LoadDatasetAsync(int dataId, CancellationToken cancellationToken = default)
        => DatasetCsv.Parse(await GetCsvAsync(dataId, cancellationToken).ConfigureAwait(false));

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses an OpenML <c>data/list</c> JSON response into dataset entries (offline).</summary>
    public static IReadOnlyList<OpenMLDatasetInfo> ParseIndex(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);

        var datasets = new List<OpenMLDatasetInfo>();
        if (document.RootElement.TryGetProperty("data", out JsonElement data)
            && data.TryGetProperty("dataset", out JsonElement list)
            && list.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement node in list.EnumerateArray())
            {
                int? id = GetInt(node, "did");
                string? name = GetString(node, "name");
                if (id is null || name is null) continue;
                datasets.Add(new OpenMLDatasetInfo(id.Value, name, GetString(node, "status") ?? ""));
            }
        }
        return datasets;
    }

    private static string? GetString(JsonElement parent, string name)
        => parent.TryGetProperty(name, out JsonElement e) && e.ValueKind == JsonValueKind.String
            ? e.GetString()
            : null;

    private static int? GetInt(JsonElement parent, string name)
    {
        if (!parent.TryGetProperty(name, out JsonElement e)) return null;
        if (e.ValueKind == JsonValueKind.Number && e.TryGetInt32(out int n)) return n;
        if (e.ValueKind == JsonValueKind.String
            && int.TryParse(e.GetString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out int s)) return s;
        return null;
    }
}
