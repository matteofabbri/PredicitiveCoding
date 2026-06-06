using PredictiveCoding.Data.Wikidata;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Offline tests for <see cref="WikidataSource"/>: ID normalisation, JSON parsing
/// (entity, search, SPARQL) and argument validation. No network is used — sample
/// JSON is parsed in memory, so the tests are deterministic.
/// </summary>
public sealed class WikidataSourceTests
{
    private const string EntityJson = """
        {
          "entities": {
            "Q42": {
              "type": "item",
              "id": "Q42",
              "labels": {
                "en": { "language": "en", "value": "Douglas Adams" },
                "it": { "language": "it", "value": "Douglas Adams" }
              },
              "descriptions": {
                "en": { "language": "en", "value": "English writer and humorist" }
              },
              "aliases": {
                "en": [
                  { "language": "en", "value": "Douglas Noel Adams" },
                  { "language": "en", "value": "Douglas Noël Adams" }
                ]
              },
              "claims": {
                "P31": [ {
                  "mainsnak": {
                    "snaktype": "value", "property": "P31",
                    "datavalue": { "value": { "entity-type": "item", "numeric-id": 5, "id": "Q5" }, "type": "wikibase-entityid" },
                    "datatype": "wikibase-item"
                  },
                  "rank": "normal"
                } ],
                "P569": [ {
                  "mainsnak": {
                    "snaktype": "value", "property": "P569",
                    "datavalue": { "value": { "time": "+1952-03-11T00:00:00Z", "precision": 11 }, "type": "time" },
                    "datatype": "time"
                  },
                  "rank": "normal"
                } ]
              },
              "sitelinks": {
                "enwiki": { "site": "enwiki", "title": "Douglas Adams" }
              }
            }
          }
        }
        """;

    private const string SearchJson = """
        {
          "searchinfo": { "search": "douglas adams" },
          "search": [
            { "id": "Q42", "label": "Douglas Adams", "description": "English writer", "url": "//www.wikidata.org/wiki/Q42" }
          ],
          "success": 1
        }
        """;

    private const string SparqlJson = """
        {
          "head": { "vars": [ "item", "itemLabel" ] },
          "results": {
            "bindings": [
              {
                "item": { "type": "uri", "value": "http://www.wikidata.org/entity/Q42" },
                "itemLabel": { "type": "literal", "value": "Douglas Adams", "xml:lang": "en" }
              }
            ]
          }
        }
        """;

    // ── ID normalisation ─────────────────────────────────────────────────────

    [Theory]
    [InlineData("Q42", "Q42")]
    [InlineData("q42", "Q42")]
    [InlineData(" p31 ", "P31")]
    [InlineData("http://www.wikidata.org/entity/Q42", "Q42")]
    [InlineData("L1234", "L1234")]
    public void NormalizeEntityId_AcceptsValidIds(string input, string expected)
        => Assert.Equal(expected, WikidataSource.NormalizeEntityId(input));

    [Theory]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("42")]
    [InlineData("Q")]
    [InlineData("XYZ")]
    [InlineData("Q4a2")]
    public void NormalizeEntityId_RejectsInvalidIds(string input)
        => Assert.Throws<ArgumentException>(() => WikidataSource.NormalizeEntityId(input));

    [Fact]
    public void NormalizeEntityId_RejectsNull()
        => Assert.Throws<ArgumentNullException>(() => WikidataSource.NormalizeEntityId(null!));

    // ── Entity parsing ───────────────────────────────────────────────────────

    [Fact]
    public void ParseEntity_ReadsLabelsDescriptionsAndAliases()
    {
        WikidataEntity entity = WikidataSource.ParseEntity(EntityJson);

        Assert.Equal("Q42", entity.Id);
        Assert.Equal("Douglas Adams", entity.Label());          // default "en"
        Assert.Equal("Douglas Adams", entity.Label("it"));
        Assert.Null(entity.Label("fr"));
        Assert.Contains("writer", entity.Description()!);
        Assert.Contains("Douglas Noel Adams", entity.Aliases["en"]);
    }

    [Fact]
    public void ParseEntity_ReadsClaimsAndSitelinks()
    {
        WikidataEntity entity = WikidataSource.ParseEntity(EntityJson);

        WikidataStatement instanceOf = Assert.Single(entity.Claims["P31"]);
        Assert.Equal("wikibase-entityid", instanceOf.Value.Type);
        Assert.Equal("Q5", instanceOf.Value.EntityId);

        WikidataStatement birth = Assert.Single(entity.Claims["P569"]);
        Assert.Equal("time", birth.Value.Type);
        Assert.Equal("+1952-03-11T00:00:00Z", birth.Value.Text);

        Assert.Equal("Douglas Adams", entity.Sitelinks["enwiki"]);
    }

    [Fact]
    public void ParseEntity_RejectsInvalidJson()
    {
        Assert.Throws<ArgumentException>(() => WikidataSource.ParseEntity(""));
        Assert.Throws<FormatException>(() => WikidataSource.ParseEntity("{ \"foo\": 1 }"));
    }

    // ── Search parsing ───────────────────────────────────────────────────────

    [Fact]
    public void ParseSearchResults_ReadsHits()
    {
        IReadOnlyList<WikidataSearchResult> hits = WikidataSource.ParseSearchResults(SearchJson);

        WikidataSearchResult hit = Assert.Single(hits);
        Assert.Equal("Q42", hit.Id);
        Assert.Equal("Douglas Adams", hit.Label);
        Assert.Equal("English writer", hit.Description);
    }

    // ── SPARQL parsing ───────────────────────────────────────────────────────

    [Fact]
    public void ParseSparqlResults_ReadsVariablesAndBindings()
    {
        SparqlResults results = WikidataSource.ParseSparqlResults(SparqlJson);

        Assert.Equal(new[] { "item", "itemLabel" }, results.Variables);
        IReadOnlyDictionary<string, SparqlValue> row = Assert.Single(results.Rows);
        Assert.Equal("Q42", row["item"].EntityId);
        Assert.Equal("Douglas Adams", row["itemLabel"].Value);
        Assert.Equal("en", row["itemLabel"].Language);
        Assert.Null(row["itemLabel"].EntityId); // a literal, not an entity URI
    }

    // ── Construction & validation ────────────────────────────────────────────

    [Fact]
    public void Dispose_DoesNotDisposeCallerSuppliedClient()
    {
        using var http = new HttpClient();
        new WikidataSource(http).Dispose();
        // If the source had disposed the caller's client, this setter would throw.
        http.Timeout = TimeSpan.FromSeconds(10);
        Assert.Equal(TimeSpan.FromSeconds(10), http.Timeout);
    }

    [Fact]
    public void Dispose_OwnedClient_IsIdempotent()
    {
        var source = new WikidataSource();
        source.Dispose();
        source.Dispose(); // must not throw
    }

    [Fact]
    public async Task AsyncMethods_ValidateBeforeNetwork()
    {
        using var source = new WikidataSource();
        await Assert.ThrowsAsync<ArgumentException>(() => source.GetEntityAsync("not-an-id"));
        await Assert.ThrowsAsync<ArgumentException>(() => source.SearchAsync(""));
        await Assert.ThrowsAsync<ArgumentException>(() => source.QueryAsync(""));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(() => source.SearchAsync("x", "en", 0));
    }
}
