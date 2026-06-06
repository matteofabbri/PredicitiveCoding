using System.Text.Json;

namespace PredictiveCoding.Data.Datasets;

/// <summary>
/// The <b>Python datasets</b> source, backed by the <i>seaborn-data</i> repository
/// (<c>https://github.com/mwaskom/seaborn-data</c>) — the small canonical datasets
/// loaded across the Python ecosystem via <c>seaborn.load_dataset(name)</c>
/// (<c>iris</c>, <c>tips</c>, <c>titanic</c>, <c>penguins</c>, <c>flights</c>, …),
/// the de-facto Python analogue of R's <c>datasets</c> package. The list of
/// datasets is discovered through the GitHub trees API and each one is served as
/// CSV. JSON/CSV parsing is static and runs offline; mixed-type rows are returned
/// as a <see cref="DatasetTable"/>.
/// </summary>
public sealed class SeabornSource : IDisposable
{
    /// <summary>Base URL serving the raw CSV files.</summary>
    public const string BaseUrl = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master";

    /// <summary>GitHub trees API listing the repository contents.</summary>
    public const string TreesUrl = "https://api.github.com/repos/mwaskom/seaborn-data/git/trees/master";

    /// <summary>The well-known datasets, offered as an offline fallback to <see cref="ListDatasetsAsync"/>.</summary>
    public static IReadOnlyList<string> WellKnownDatasets { get; } =
    [
        "anagrams", "anscombe", "attention", "brain_networks", "car_crashes", "diamonds",
        "dots", "dowjones", "exercise", "flights", "fmri", "geyser", "glue", "healthexp",
        "iris", "mpg", "penguins", "planets", "seaice", "taxis", "tips", "titanic",
    ];

    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public SeabornSource(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Lists the dataset names available in the repository (via the GitHub trees API).</summary>
    public async Task<IReadOnlyList<string>> ListDatasetsAsync(CancellationToken cancellationToken = default)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, TreesUrl);
        request.Headers.UserAgent.ParseAdd("PredictiveCoding.Data"); // GitHub API requires a User-Agent
        request.Headers.Accept.ParseAdd("application/vnd.github+json");
        using HttpResponseMessage response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return ParseIndex(await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false));
    }

    /// <summary>Downloads the raw CSV of one dataset (e.g. <c>penguins</c>).</summary>
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

    /// <summary>Parses a GitHub trees response, returning the dataset names (CSV blobs, without the extension) (offline).</summary>
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
