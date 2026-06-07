using System.Text.Json;

namespace PredictiveCoding.Data.Datasets;

/// <summary>
/// The <b>Plotly datasets</b> source, backed by the <i>plotly/datasets</i> repository
/// (<c>https://github.com/plotly/datasets</c>) — the example CSVs used throughout
/// Plotly's documentation (<c>iris</c>, <c>tips</c>, <c>gapminderDataFiveYear</c>,
/// <c>auto-mpg</c>, <c>diabetes</c>, …), stored flat at the repository root. The list
/// is discovered through the GitHub trees API and each dataset is served as CSV.
/// JSON/CSV parsing is static and runs offline; mixed-type rows are returned as a
/// <see cref="DatasetTable"/>.
/// </summary>
public sealed class PlotlyDatasetsSource : IDisposable
{
    /// <summary>Base URL serving the raw CSV files.</summary>
    public const string BaseUrl = "https://raw.githubusercontent.com/plotly/datasets/master";

    /// <summary>GitHub trees API listing the repository root.</summary>
    public const string TreesUrl = "https://api.github.com/repos/plotly/datasets/git/trees/master";

    /// <summary>Well-known datasets, offered as an offline fallback to <see cref="ListDatasetsAsync"/>.</summary>
    public static IReadOnlyList<string> WellKnownDatasets { get; } =
    [
        "2010_alcohol_consumption_by_country", "2011_us_ag_exports", "2014_apple_stock",
        "2014_usa_states", "auto-mpg", "diabetes", "gapminderDataFiveYear", "iris", "tips",
        "volcano_db", "wind_speed_laurel_nebraska", "wine_data",
    ];

    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public PlotlyDatasetsSource(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Lists the CSV dataset names available at the repository root.</summary>
    public async Task<IReadOnlyList<string>> ListDatasetsAsync(CancellationToken cancellationToken = default)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, TreesUrl);
        request.Headers.UserAgent.ParseAdd("PredictiveCoding.Data"); // GitHub API requires a User-Agent
        request.Headers.Accept.ParseAdd("application/vnd.github+json");
        using HttpResponseMessage response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return ParseIndex(await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Downloads the raw CSV of one dataset (e.g. <c>iris</c>).</summary>
    public async Task<string> GetCsvAsync(string name, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        string url = $"{BaseUrl}/{Uri.EscapeDataString(name)}.csv";
        return await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Downloads one dataset and parses it into a string-typed table.</summary>
    public async Task<DatasetTable> LoadDatasetAsync(string name, CancellationToken cancellationToken = default)
        => DatasetCsv.Parse(await GetCsvAsync(name, cancellationToken).ConfigureAwait(false));

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses a GitHub trees response, returning the root-level CSV dataset names
    /// (without the <c>.csv</c> extension) (offline).</summary>
    public static IReadOnlyList<string> ParseIndex(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);

        var names = new List<string>();
        if (document.RootElement.TryGetProperty("tree", out JsonElement tree) && tree.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement node in tree.EnumerateArray())
            {
                string? type = GetString(node, "type");
                string? path = GetString(node, "path");
                if (type == "blob" && path is not null && path.EndsWith(".csv", StringComparison.OrdinalIgnoreCase))
                    names.Add(path[..^4]);
            }
        }
        names.Sort(StringComparer.OrdinalIgnoreCase);
        return names;
    }

    private static string? GetString(JsonElement parent, string name)
        => parent.TryGetProperty(name, out JsonElement element) && element.ValueKind == JsonValueKind.String
            ? element.GetString()
            : null;
}
