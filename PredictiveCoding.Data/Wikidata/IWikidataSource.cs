namespace PredictiveCoding.Data.Wikidata;

/// <summary>
/// Reads structured data from Wikidata: entities by ID, entity search and
/// SPARQL queries against the Wikidata Query Service.
/// </summary>
public interface IWikidataSource
{
    /// <summary>Fetches a full entity by its ID (e.g. <c>Q42</c>, <c>P31</c>).</summary>
    Task<WikidataEntity> GetEntityAsync(string id, CancellationToken cancellationToken = default);

    /// <summary>Searches entities by free text.</summary>
    Task<IReadOnlyList<WikidataSearchResult>> SearchAsync(
        string query, string language = "en", int limit = 10, CancellationToken cancellationToken = default);

    /// <summary>Runs a SPARQL query against the Wikidata Query Service.</summary>
    Task<SparqlResults> QueryAsync(string sparql, CancellationToken cancellationToken = default);
}
