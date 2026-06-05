using System.Text.Json;

namespace PredictiveCoding.Data.HuggingFace;

/// <summary>
/// Data source for the Hugging Face Hub datasets via the datasets-server "rows"
/// API. JSON parsing is static and runs offline. Each row is returned as a
/// column-keyed dictionary of primitive values.
/// </summary>
public sealed class HuggingFaceDatasets : IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public HuggingFaceDatasets(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Fetches a window of rows from a dataset split.</summary>
    public async Task<IReadOnlyList<IReadOnlyDictionary<string, object?>>> GetRowsAsync(
        string dataset, string config = "default", string split = "train",
        int offset = 0, int length = 100, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(dataset);
        ArgumentException.ThrowIfNullOrWhiteSpace(config);
        ArgumentException.ThrowIfNullOrWhiteSpace(split);
        if (offset < 0) throw new ArgumentOutOfRangeException(nameof(offset));
        if (length <= 0) throw new ArgumentOutOfRangeException(nameof(length));

        string url = $"https://datasets-server.huggingface.co/rows?dataset={Uri.EscapeDataString(dataset)}"
                   + $"&config={Uri.EscapeDataString(config)}&split={Uri.EscapeDataString(split)}"
                   + $"&offset={offset}&length={length}";
        return ParseRows(await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses a datasets-server rows response (offline).</summary>
    public static IReadOnlyList<IReadOnlyDictionary<string, object?>> ParseRows(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);

        var rows = new List<IReadOnlyDictionary<string, object?>>();
        if (document.RootElement.TryGetProperty("rows", out JsonElement array) && array.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement entry in array.EnumerateArray())
            {
                if (!entry.TryGetProperty("row", out JsonElement row) || row.ValueKind != JsonValueKind.Object)
                    continue;
                var record = new Dictionary<string, object?>();
                foreach (JsonProperty field in row.EnumerateObject())
                    record[field.Name] = ToValue(field.Value);
                rows.Add(record);
            }
        }
        return rows;
    }

    private static object? ToValue(JsonElement element) => element.ValueKind switch
    {
        JsonValueKind.String => element.GetString(),
        JsonValueKind.Number => element.TryGetInt64(out long l) ? l : (object)element.GetDouble(),
        JsonValueKind.True => true,
        JsonValueKind.False => false,
        JsonValueKind.Null or JsonValueKind.Undefined => null,
        _ => element.ToString(),
    };
}
