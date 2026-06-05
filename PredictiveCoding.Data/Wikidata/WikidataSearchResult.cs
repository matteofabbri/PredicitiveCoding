namespace PredictiveCoding.Data.Wikidata;

/// <summary>
/// A single hit from a Wikidata entity search (<c>wbsearchentities</c>).
/// </summary>
public sealed record WikidataSearchResult(string Id, string? Label, string? Description, string? Url);
