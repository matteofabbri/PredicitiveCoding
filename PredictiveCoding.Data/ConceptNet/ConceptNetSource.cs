using System.Text.Json;

namespace PredictiveCoding.Data.ConceptNet;

/// <summary>A ConceptNet assertion (edge) between two concepts.</summary>
public sealed record ConceptNetEdge(string Start, string End, string Relation, double Weight);

/// <summary>
/// Commonsense-knowledge source backed by the ConceptNet API. JSON parsing is
/// static and runs offline.
/// </summary>
public sealed class ConceptNetSource : IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public ConceptNetSource(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Fetches the edges of a concept (e.g. <c>dog</c>).</summary>
    public async Task<IReadOnlyList<ConceptNetEdge>> GetEdgesAsync(
        string term, string language = "en", int limit = 50, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(term);
        ArgumentException.ThrowIfNullOrWhiteSpace(language);
        if (limit <= 0) throw new ArgumentOutOfRangeException(nameof(limit));
        string url = $"https://api.conceptnet.io/c/{Uri.EscapeDataString(language)}/{Uri.EscapeDataString(term)}?limit={limit}";
        return ParseEdges(await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses a ConceptNet response (offline).</summary>
    public static IReadOnlyList<ConceptNetEdge> ParseEdges(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);

        var edges = new List<ConceptNetEdge>();
        if (document.RootElement.TryGetProperty("edges", out JsonElement array) && array.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement edge in array.EnumerateArray())
            {
                edges.Add(new ConceptNetEdge(
                    Start: Label(edge, "start"),
                    End: Label(edge, "end"),
                    Relation: Label(edge, "rel"),
                    Weight: edge.TryGetProperty("weight", out var w) && w.ValueKind == JsonValueKind.Number ? w.GetDouble() : 0.0));
            }
        }
        return edges;
    }

    private static string Label(JsonElement edge, string property)
        => edge.TryGetProperty(property, out var node) && node.TryGetProperty("label", out var label) &&
           label.ValueKind == JsonValueKind.String
            ? label.GetString()!
            : string.Empty;
}
