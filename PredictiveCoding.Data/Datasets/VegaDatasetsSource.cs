using System.Text.Json;

namespace PredictiveCoding.Data.Datasets;

/// <summary>
/// The <b>Vega datasets</b> source, backed by the <i>vega-datasets</i> repository
/// (<c>https://github.com/vega/vega-datasets</c>) — the example datasets used across
/// Vega, Vega-Lite and Altair (<c>cars</c>, <c>movies</c>, <c>seattle-weather</c>,
/// <c>stocks</c>, <c>gapminder</c>, …). The CSV files live under the repo's
/// <c>data/</c> folder; the list is discovered through the GitHub trees API and each
/// dataset is served as CSV. JSON/CSV parsing is static and runs offline; mixed-type
/// rows are returned as a <see cref="DatasetTable"/>.
/// </summary>
public sealed class VegaDatasetsSource : IDisposable
{
    /// <summary>Base URL serving the raw CSV files (the repo's <c>data/</c> folder).</summary>
    public const string BaseUrl = "https://raw.githubusercontent.com/vega/vega-datasets/main/data";

    /// <summary>GitHub trees API listing the repository recursively.</summary>
    public const string TreesUrl = "https://api.github.com/repos/vega/vega-datasets/git/trees/main?recursive=1";

    /// <summary>Well-known datasets, offered as an offline fallback to <see cref="ListDatasetsAsync"/>.</summary>
    public static IReadOnlyList<string> WellKnownDatasets { get; } =
    [
        "airports", "cars", "co2-concentration", "disasters", "gapminder", "iowa-electricity",
        "iris", "jobs", "movies", "penguins", "population", "seattle-weather", "sp500",
        "stocks", "us-employment", "weather", "wheat", "zipcodes",
    ];

    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public VegaDatasetsSource(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Lists the CSV dataset names available under the repo's <c>data/</c> folder.</summary>
    public async Task<IReadOnlyList<string>> ListDatasetsAsync(CancellationToken cancellationToken = default)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, TreesUrl);
        request.Headers.UserAgent.ParseAdd("PredictiveCoding.Data"); // GitHub API requires a User-Agent
        request.Headers.Accept.ParseAdd("application/vnd.github+json");
        using HttpResponseMessage response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return ParseIndex(await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Downloads the raw CSV of one dataset (e.g. <c>cars</c>).</summary>
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

    /// <summary>Parses a recursive GitHub trees response, returning the CSV dataset names
    /// under <c>data/</c> (without the folder prefix or the <c>.csv</c> extension) (offline).</summary>
    public static IReadOnlyList<string> ParseIndex(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);

        var names = new List<string>();
        if (document.RootElement.TryGetProperty("tree", out JsonElement tree) && tree.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement node in tree.EnumerateArray())
            {
                if (GetString(node, "type") != "blob") continue;
                string? path = GetString(node, "path");
                if (path is not null
                    && path.StartsWith("data/", StringComparison.OrdinalIgnoreCase)
                    && path.EndsWith(".csv", StringComparison.OrdinalIgnoreCase))
                    names.Add(path["data/".Length..^4]);
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
