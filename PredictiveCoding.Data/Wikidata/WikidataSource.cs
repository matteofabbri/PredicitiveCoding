using System.Text.Json;

namespace PredictiveCoding.Data.Wikidata;

/// <summary>
/// Wikidata data source built directly on the public HTTP APIs (no third-party
/// dependency): the entity-data endpoint, <c>wbsearchentities</c> search and the
/// SPARQL Query Service. JSON is parsed with <see cref="System.Text.Json"/>.
/// </summary>
/// <remarks>
/// The network methods require internet access; the static parsing and
/// ID-normalisation helpers run offline.
/// </remarks>
public sealed class WikidataSource : IWikidataSource, IDisposable
{
    private const string DefaultUserAgent = "PredictiveCoding.Data/1.0 (Wikidata data source)";
    private const string EntityDataUrl = "https://www.wikidata.org/wiki/Special:EntityData/";
    private const string ApiUrl = "https://www.wikidata.org/w/api.php";
    private const string SparqlUrl = "https://query.wikidata.org/sparql";

    private readonly HttpClient _http;
    private readonly bool _ownsHttp;
    private readonly string _userAgent;

    /// <summary>
    /// Creates a source. When <paramref name="httpClient"/> is null an internal
    /// client is created and disposed with this instance. Wikimedia requires a
    /// descriptive <paramref name="userAgent"/>.
    /// </summary>
    public WikidataSource(HttpClient? httpClient = null, string? userAgent = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
        _userAgent = string.IsNullOrWhiteSpace(userAgent) ? DefaultUserAgent : userAgent;
    }

    // ── Network ──────────────────────────────────────────────────────────────

    /// <inheritdoc />
    public async Task<WikidataEntity> GetEntityAsync(string id, CancellationToken cancellationToken = default)
    {
        string normalized = NormalizeEntityId(id);
        string url = $"{EntityDataUrl}{normalized}.json";
        string json = await GetStringAsync(url, cancellationToken).ConfigureAwait(false);
        return ParseEntity(json);
    }

    /// <inheritdoc />
    public async Task<IReadOnlyList<WikidataSearchResult>> SearchAsync(
        string query, string language = "en", int limit = 10, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(query);
        ArgumentException.ThrowIfNullOrWhiteSpace(language);
        if (limit <= 0)
            throw new ArgumentOutOfRangeException(nameof(limit), limit, "Limit must be positive.");

        string url = $"{ApiUrl}?action=wbsearchentities&format=json&type=item"
                   + $"&language={Uri.EscapeDataString(language)}&uselang={Uri.EscapeDataString(language)}"
                   + $"&limit={limit}&search={Uri.EscapeDataString(query)}";
        string json = await GetStringAsync(url, cancellationToken).ConfigureAwait(false);
        return ParseSearchResults(json);
    }

    /// <inheritdoc />
    public async Task<SparqlResults> QueryAsync(string sparql, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sparql);
        string url = $"{SparqlUrl}?format=json&query={Uri.EscapeDataString(sparql)}";
        string json = await GetStringAsync(url, cancellationToken).ConfigureAwait(false);
        return ParseSparqlResults(json);
    }

    private async Task<string> GetStringAsync(string url, CancellationToken cancellationToken)
    {
        using var request = new HttpRequestMessage(HttpMethod.Get, url);
        request.Headers.UserAgent.ParseAdd(_userAgent);
        request.Headers.Accept.ParseAdd("application/json");

        using HttpResponseMessage response = await _http.SendAsync(request, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    // ── ID normalisation (offline) ───────────────────────────────────────────

    /// <summary>
    /// Validates and normalises a Wikidata entity ID. Accepts a bare id
    /// (<c>q42</c>) or a full entity URI, returning the upper-cased id (<c>Q42</c>).
    /// </summary>
    public static string NormalizeEntityId(string id)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(id);

        string trimmed = id.Trim();
        int slash = trimmed.LastIndexOf('/');
        if (slash >= 0)
            trimmed = trimmed[(slash + 1)..];

        string upper = trimmed.ToUpperInvariant();
        if (!IsValidEntityId(upper))
            throw new ArgumentException(
                $"Invalid Wikidata entity ID: '{id}'. Expected a Q/P/L id such as 'Q42'.", nameof(id));
        return upper;
    }

    private static bool IsValidEntityId(string id)
    {
        if (id.Length < 2 || (id[0] != 'Q' && id[0] != 'P' && id[0] != 'L'))
            return false;
        for (int i = 1; i < id.Length; i++)
            if (!char.IsAsciiDigit(id[i]))
                return false;
        return true;
    }

    // ── Parsing (offline) ────────────────────────────────────────────────────

    /// <summary>Parses the JSON returned by the entity-data endpoint into a <see cref="WikidataEntity"/>.</summary>
    public static WikidataEntity ParseEntity(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);

        if (!document.RootElement.TryGetProperty("entities", out JsonElement entities) ||
            entities.ValueKind != JsonValueKind.Object)
            throw new FormatException("Invalid Wikidata entity JSON: missing 'entities' object.");

        JsonElement entity = default;
        bool found = false;
        foreach (JsonProperty property in entities.EnumerateObject())
        {
            entity = property.Value;
            found = true;
            break;
        }
        if (!found)
            throw new FormatException("Invalid Wikidata entity JSON: no entity present.");

        string id = GetString(entity, "id") ?? string.Empty;
        return new WikidataEntity(
            id,
            ParseLanguageMap(entity, "labels"),
            ParseLanguageMap(entity, "descriptions"),
            ParseAliases(entity),
            ParseClaims(entity),
            ParseSitelinks(entity));
    }

    /// <summary>Parses the JSON returned by <c>wbsearchentities</c>.</summary>
    public static IReadOnlyList<WikidataSearchResult> ParseSearchResults(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);

        var results = new List<WikidataSearchResult>();
        if (document.RootElement.TryGetProperty("search", out JsonElement search) &&
            search.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement hit in search.EnumerateArray())
            {
                results.Add(new WikidataSearchResult(
                    Id: GetString(hit, "id") ?? string.Empty,
                    Label: GetString(hit, "label"),
                    Description: GetString(hit, "description"),
                    Url: GetString(hit, "url") ?? GetString(hit, "concepturi")));
            }
        }
        return results;
    }

    /// <summary>Parses the JSON returned by the SPARQL Query Service (<c>format=json</c>).</summary>
    public static SparqlResults ParseSparqlResults(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        var variables = new List<string>();
        if (root.TryGetProperty("head", out JsonElement head) &&
            head.TryGetProperty("vars", out JsonElement vars) &&
            vars.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement v in vars.EnumerateArray())
                if (v.ValueKind == JsonValueKind.String)
                    variables.Add(v.GetString()!);
        }

        var rows = new List<IReadOnlyDictionary<string, SparqlValue>>();
        if (root.TryGetProperty("results", out JsonElement results) &&
            results.TryGetProperty("bindings", out JsonElement bindings) &&
            bindings.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement binding in bindings.EnumerateArray())
            {
                var row = new Dictionary<string, SparqlValue>();
                foreach (JsonProperty cell in binding.EnumerateObject())
                {
                    row[cell.Name] = new SparqlValue(
                        Type: GetString(cell.Value, "type") ?? string.Empty,
                        Value: GetString(cell.Value, "value") ?? string.Empty,
                        Language: GetString(cell.Value, "xml:lang"),
                        DataType: GetString(cell.Value, "datatype"));
                }
                rows.Add(row);
            }
        }
        return new SparqlResults(variables, rows);
    }

    // ── Parsing helpers ──────────────────────────────────────────────────────

    private static IReadOnlyDictionary<string, string> ParseLanguageMap(JsonElement entity, string name)
    {
        var map = new Dictionary<string, string>();
        if (entity.TryGetProperty(name, out JsonElement element) && element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                string? value = GetString(property.Value, "value");
                if (value is not null)
                    map[property.Name] = value;
            }
        }
        return map;
    }

    private static IReadOnlyDictionary<string, IReadOnlyList<string>> ParseAliases(JsonElement entity)
    {
        var map = new Dictionary<string, IReadOnlyList<string>>();
        if (entity.TryGetProperty("aliases", out JsonElement element) && element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                var list = new List<string>();
                if (property.Value.ValueKind == JsonValueKind.Array)
                {
                    foreach (JsonElement alias in property.Value.EnumerateArray())
                    {
                        string? value = GetString(alias, "value");
                        if (value is not null)
                            list.Add(value);
                    }
                }
                map[property.Name] = list;
            }
        }
        return map;
    }

    private static IReadOnlyDictionary<string, IReadOnlyList<WikidataStatement>> ParseClaims(JsonElement entity)
    {
        var map = new Dictionary<string, IReadOnlyList<WikidataStatement>>();
        if (entity.TryGetProperty("claims", out JsonElement claims) && claims.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in claims.EnumerateObject())
            {
                var statements = new List<WikidataStatement>();
                if (property.Value.ValueKind == JsonValueKind.Array)
                {
                    foreach (JsonElement statement in property.Value.EnumerateArray())
                    {
                        string? rank = GetString(statement, "rank");
                        statements.Add(new WikidataStatement(property.Name, ParseSnak(statement), rank));
                    }
                }
                map[property.Name] = statements;
            }
        }
        return map;
    }

    private static WikidataValue ParseSnak(JsonElement statement)
    {
        if (!statement.TryGetProperty("mainsnak", out JsonElement snak))
            return new WikidataValue("unknown", null, null);

        string snakType = GetString(snak, "snaktype") ?? "value";
        if (snakType != "value")
            return new WikidataValue(snakType, null, null); // novalue / somevalue

        if (!snak.TryGetProperty("datavalue", out JsonElement dataValue))
            return new WikidataValue("unknown", null, null);

        string type = GetString(dataValue, "type") ?? "unknown";
        JsonElement value = dataValue.TryGetProperty("value", out JsonElement v) ? v : default;
        if (value.ValueKind == JsonValueKind.Undefined)
            return new WikidataValue(type, null, null);

        return type switch
        {
            "wikibase-entityid" => new WikidataValue(type, GetString(value, "id"), GetString(value, "id")),
            "string" => new WikidataValue(type, AsText(value), null),
            "monolingualtext" => new WikidataValue(type, GetString(value, "text"), null),
            "time" => new WikidataValue(type, GetString(value, "time"), null),
            "quantity" => new WikidataValue(type, GetString(value, "amount"), null),
            "globecoordinate" => new WikidataValue(type, GlobeText(value), null),
            _ => new WikidataValue(type, AsText(value), null),
        };
    }

    private static string? GlobeText(JsonElement value)
    {
        bool hasLat = value.TryGetProperty("latitude", out JsonElement lat) && lat.ValueKind == JsonValueKind.Number;
        bool hasLon = value.TryGetProperty("longitude", out JsonElement lon) && lon.ValueKind == JsonValueKind.Number;
        return hasLat && hasLon ? $"{lat.GetDouble()},{lon.GetDouble()}" : null;
    }

    private static IReadOnlyDictionary<string, string> ParseSitelinks(JsonElement entity)
    {
        var map = new Dictionary<string, string>();
        if (entity.TryGetProperty("sitelinks", out JsonElement element) && element.ValueKind == JsonValueKind.Object)
        {
            foreach (JsonProperty property in element.EnumerateObject())
            {
                string? title = GetString(property.Value, "title");
                if (title is not null)
                    map[property.Name] = title;
            }
        }
        return map;
    }

    // Reads a named string-ish property (returns null when absent; renders non-strings as raw JSON).
    private static string? GetString(JsonElement parent, string name)
    {
        if (parent.ValueKind != JsonValueKind.Object || !parent.TryGetProperty(name, out JsonElement element))
            return null;
        return AsText(element);
    }

    private static string? AsText(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.String => element.GetString(),
        JsonValueKind.Undefined or JsonValueKind.Null => null,
        _ => element.ToString(),
    };
}
