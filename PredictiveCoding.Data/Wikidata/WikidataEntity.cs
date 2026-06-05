namespace PredictiveCoding.Data.Wikidata;

/// <summary>
/// A normalised value of a Wikidata statement (the main snak).
/// </summary>
/// <param name="Type">
/// The value type: <c>wikibase-entityid</c>, <c>string</c>, <c>monolingualtext</c>,
/// <c>time</c>, <c>quantity</c>, <c>globecoordinate</c>, or <c>novalue</c>/<c>somevalue</c>.
/// </param>
/// <param name="Text">A human-readable rendering of the value (null when absent).</param>
/// <param name="EntityId">The referenced Q/P/L id when <see cref="Type"/> is <c>wikibase-entityid</c>.</param>
public sealed record WikidataValue(string Type, string? Text, string? EntityId)
{
    public override string ToString() => Text ?? EntityId ?? Type;
}

/// <summary>A single statement (claim) about an entity.</summary>
public sealed record WikidataStatement(string Property, WikidataValue Value, string? Rank);

/// <summary>
/// A Wikidata entity (item <c>Q…</c>, property <c>P…</c> or lexeme <c>L…</c>),
/// normalised across the API's verbose JSON.
/// </summary>
public sealed record WikidataEntity(
    string Id,
    IReadOnlyDictionary<string, string> Labels,
    IReadOnlyDictionary<string, string> Descriptions,
    IReadOnlyDictionary<string, IReadOnlyList<string>> Aliases,
    IReadOnlyDictionary<string, IReadOnlyList<WikidataStatement>> Claims,
    IReadOnlyDictionary<string, string> Sitelinks)
{
    /// <summary>The label in <paramref name="language"/>, or null.</summary>
    public string? Label(string language = "en") => Labels.TryGetValue(language, out string? v) ? v : null;

    /// <summary>The description in <paramref name="language"/>, or null.</summary>
    public string? Description(string language = "en") => Descriptions.TryGetValue(language, out string? v) ? v : null;
}
