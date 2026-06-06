using System.Globalization;
using CsvHelper;
using CsvHelper.Configuration;

namespace PredictiveCoding.Data.Datasets;

/// <summary>One entry of the Rdatasets catalogue (a single dataset shipped by an R package).</summary>
/// <param name="Package">The R package the dataset belongs to (e.g. <c>datasets</c>, <c>AER</c>).</param>
/// <param name="Item">The dataset name within the package (e.g. <c>iris</c>).</param>
/// <param name="Title">A human-readable description.</param>
/// <param name="Rows">Row count reported by the catalogue.</param>
/// <param name="Cols">Column count reported by the catalogue.</param>
/// <param name="CsvUrl">Absolute URL of the dataset as CSV.</param>
/// <param name="DocUrl">Absolute URL of the dataset's HTML documentation.</param>
public sealed record RDatasetInfo(
    string Package, string Item, string Title, int Rows, int Cols, string CsvUrl, string DocUrl);

/// <summary>
/// The <b>R datasets</b> source, backed by Vincent Arel-Bundock's <i>Rdatasets</i>
/// mirror (<c>https://vincentarelbundock.github.io/Rdatasets/</c>) — the canonical
/// collection of the datasets bundled with R's <c>datasets</c> package and ~50
/// other packages (<c>iris</c>, <c>mtcars</c>, <c>AirPassengers</c>, …), served as
/// CSV with a single index file. Catalogue and CSV parsing are static and run
/// offline; mixed-type rows are returned as a <see cref="DatasetTable"/>.
/// </summary>
public sealed class RDatasetsSource : IDisposable
{
    /// <summary>Base URL of the Rdatasets mirror.</summary>
    public const string BaseUrl = "https://vincentarelbundock.github.io/Rdatasets";

    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public RDatasetsSource(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Fetches the full catalogue of available datasets (~2000 entries).</summary>
    public async Task<IReadOnlyList<RDatasetInfo>> ListDatasetsAsync(CancellationToken cancellationToken = default)
        => ParseIndex(await _http.GetStringAsync($"{BaseUrl}/datasets.csv", cancellationToken).ConfigureAwait(false));

    /// <summary>Downloads the raw CSV of one dataset (e.g. <c>datasets</c>, <c>iris</c>).</summary>
    public async Task<string> GetCsvAsync(string package, string item, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(package);
        ArgumentException.ThrowIfNullOrWhiteSpace(item);
        string url = $"{BaseUrl}/csv/{Uri.EscapeDataString(package)}/{Uri.EscapeDataString(item)}.csv";
        return await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Downloads one dataset and parses it into a string-typed table.</summary>
    public async Task<DatasetTable> LoadDatasetAsync(string package, string item, CancellationToken cancellationToken = default)
        => DatasetCsv.Parse(await GetCsvAsync(package, item, cancellationToken).ConfigureAwait(false));

    /// <summary>Builds the documentation URL for a dataset (no request is made).</summary>
    public static string DocUrlFor(string package, string item)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(package);
        ArgumentException.ThrowIfNullOrWhiteSpace(item);
        return $"{BaseUrl}/doc/{Uri.EscapeDataString(package)}/{Uri.EscapeDataString(item)}.html";
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses the <c>datasets.csv</c> catalogue, mapping fields by header name (offline).</summary>
    public static IReadOnlyList<RDatasetInfo> ParseIndex(string csv)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(csv);

        var config = new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            DetectColumnCountChanges = false,
            MissingFieldFound = null,
            BadDataFound = null,
        };

        using var reader = new StringReader(csv);
        using var parser = new CsvParser(reader, config);

        if (!parser.Read() || parser.Record is null)
            throw new InvalidDataException("Rdatasets index is empty.");

        string[] header = parser.Record;
        int Col(string name) => Array.FindIndex(header, h => string.Equals(h, name, StringComparison.OrdinalIgnoreCase));
        int pkg = Col("Package"), item = Col("Item"), title = Col("Title"),
            rowsCol = Col("Rows"), colsCol = Col("Cols"), csvCol = Col("CSV"), docCol = Col("Doc");
        if (pkg < 0 || item < 0)
            throw new InvalidDataException("Rdatasets index is missing the Package/Item columns.");

        var datasets = new List<RDatasetInfo>();
        while (parser.Read())
        {
            string[] record = parser.Record ?? [];
            string Field(int i) => i >= 0 && i < record.Length ? record[i] : string.Empty;

            string package = Field(pkg), name = Field(item);
            if (package.Length == 0 || name.Length == 0)
                continue;

            int.TryParse(Field(rowsCol), NumberStyles.Integer, CultureInfo.InvariantCulture, out int nRows);
            int.TryParse(Field(colsCol), NumberStyles.Integer, CultureInfo.InvariantCulture, out int nCols);
            datasets.Add(new RDatasetInfo(package, name, Field(title), nRows, nCols, Field(csvCol), Field(docCol)));
        }
        return datasets;
    }
}
