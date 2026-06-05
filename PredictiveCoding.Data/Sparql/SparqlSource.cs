using PredictiveCoding.Data.Wikidata;

namespace PredictiveCoding.Data.Sparql;

/// <summary>
/// Generic SPARQL endpoint data source (DBpedia, Wikidata, or any endpoint that
/// speaks the SPARQL JSON results format). Reuses the SPARQL result model and
/// parser from the Wikidata source.
/// </summary>
public sealed class SparqlSource : IDisposable
{
    private const string DefaultUserAgent = "PredictiveCoding.Data/1.0 (SPARQL data source)";

    private readonly string _endpoint;
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;
    private readonly string _userAgent;

    public SparqlSource(string endpoint, HttpClient? httpClient = null, string? userAgent = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(endpoint);
        _endpoint = endpoint;
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
        _userAgent = string.IsNullOrWhiteSpace(userAgent) ? DefaultUserAgent : userAgent;
    }

    /// <summary>A source pointed at the DBpedia endpoint.</summary>
    public static SparqlSource DBpedia(HttpClient? httpClient = null)
        => new("https://dbpedia.org/sparql", httpClient);

    /// <summary>A source pointed at the Wikidata Query Service.</summary>
    public static SparqlSource Wikidata(HttpClient? httpClient = null)
        => new("https://query.wikidata.org/sparql", httpClient);

    /// <summary>Runs a SPARQL query and returns the result set.</summary>
    public async Task<SparqlResults> QueryAsync(string sparql, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sparql);
        string url = $"{_endpoint}?format=json&query={Uri.EscapeDataString(sparql)}";

        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        request.Headers.UserAgent.ParseAdd(_userAgent);
        request.Headers.Accept.ParseAdd("application/sparql-results+json");

        using HttpResponseMessage response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        string json = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
        return WikidataSource.ParseSparqlResults(json);
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }
}
