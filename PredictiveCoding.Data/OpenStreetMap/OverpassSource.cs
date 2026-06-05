using System.Text.Json;

namespace PredictiveCoding.Data.OpenStreetMap;

/// <summary>An element (node/way/relation) from an Overpass result.</summary>
public sealed record OsmElement(string Type, long Id, double? Lat, double? Lon, IReadOnlyDictionary<string, string> Tags);

/// <summary>
/// Geographic-data source backed by the OpenStreetMap Overpass API. JSON parsing
/// is static and runs offline.
/// </summary>
public sealed class OverpassSource : IDisposable
{
    private const string Endpoint = "https://overpass-api.de/api/interpreter";

    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public OverpassSource(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Runs an Overpass QL query (which should request <c>[out:json]</c>).</summary>
    public async Task<IReadOnlyList<OsmElement>> QueryAsync(string overpassQl, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(overpassQl);
        using var content = new FormUrlEncodedContent([new KeyValuePair<string, string>("data", overpassQl)]);
        using HttpResponseMessage response = await _http.PostAsync(Endpoint, content, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();
        return ParseElements(await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false));
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses an Overpass JSON result (offline).</summary>
    public static IReadOnlyList<OsmElement> ParseElements(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);

        var elements = new List<OsmElement>();
        if (document.RootElement.TryGetProperty("elements", out JsonElement array) && array.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement element in array.EnumerateArray())
            {
                var tags = new Dictionary<string, string>();
                if (element.TryGetProperty("tags", out JsonElement tagObject) && tagObject.ValueKind == JsonValueKind.Object)
                    foreach (JsonProperty tag in tagObject.EnumerateObject())
                        if (tag.Value.ValueKind == JsonValueKind.String)
                            tags[tag.Name] = tag.Value.GetString()!;

                elements.Add(new OsmElement(
                    Type: element.TryGetProperty("type", out var type) && type.ValueKind == JsonValueKind.String ? type.GetString()! : string.Empty,
                    Id: element.TryGetProperty("id", out var id) && id.ValueKind == JsonValueKind.Number ? id.GetInt64() : 0,
                    Lat: element.TryGetProperty("lat", out var lat) && lat.ValueKind == JsonValueKind.Number ? lat.GetDouble() : null,
                    Lon: element.TryGetProperty("lon", out var lon) && lon.ValueKind == JsonValueKind.Number ? lon.GetDouble() : null,
                    Tags: tags));
            }
        }
        return elements;
    }
}
