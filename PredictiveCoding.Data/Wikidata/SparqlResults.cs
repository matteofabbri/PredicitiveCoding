namespace PredictiveCoding.Data.Wikidata;

/// <summary>
/// A single cell of a SPARQL result row.
/// </summary>
/// <param name="Type">RDF term type: <c>uri</c>, <c>literal</c>, <c>bnode</c>, …</param>
/// <param name="Value">The raw value (URI string or literal lexical form).</param>
/// <param name="Language">Language tag for literals, or null.</param>
/// <param name="DataType">Datatype URI for typed literals, or null.</param>
public sealed record SparqlValue(string Type, string Value, string? Language, string? DataType)
{
    private const string EntityPrefix = "http://www.wikidata.org/entity/";

    /// <summary>The Q/P/L id when this is a Wikidata entity URI, otherwise null.</summary>
    public string? EntityId =>
        Type == "uri" && Value.StartsWith(EntityPrefix, StringComparison.Ordinal)
            ? Value[EntityPrefix.Length..]
            : null;
}

/// <summary>
/// The result set of a SPARQL <c>SELECT</c> query: the projected variables and
/// the rows of bindings.
/// </summary>
public sealed record SparqlResults(
    IReadOnlyList<string> Variables,
    IReadOnlyList<IReadOnlyDictionary<string, SparqlValue>> Rows);
