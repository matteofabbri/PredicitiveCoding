using System.Text.Json;

namespace PredictiveCoding.Data.WorldBank;

/// <summary>One observation of a World Bank indicator for a country/year.</summary>
public sealed record WorldBankPoint(string CountryIso, string Indicator, int Year, double? Value);

/// <summary>
/// Open-data source backed by the World Bank indicators API. JSON parsing is
/// static and runs offline.
/// </summary>
public sealed class WorldBankSource : IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public WorldBankSource(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Fetches an indicator series for a country (e.g. <c>US</c>, <c>NY.GDP.MKTP.CD</c>).</summary>
    public async Task<IReadOnlyList<WorldBankPoint>> GetIndicatorAsync(
        string countryIso, string indicator, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(countryIso);
        ArgumentException.ThrowIfNullOrWhiteSpace(indicator);
        string url = $"https://api.worldbank.org/v2/country/{Uri.EscapeDataString(countryIso)}"
                   + $"/indicator/{Uri.EscapeDataString(indicator)}?format=json&per_page=1000";
        return ParseIndicator(await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses a World Bank indicator response (offline).</summary>
    public static IReadOnlyList<WorldBankPoint> ParseIndicator(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;
        // The response is [ <metadata>, [ <points...> ] ].
        if (root.ValueKind != JsonValueKind.Array || root.GetArrayLength() < 2)
            return [];

        JsonElement data = root[1];
        var points = new List<WorldBankPoint>();
        if (data.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement point in data.EnumerateArray())
            {
                string indicator = point.TryGetProperty("indicator", out var ind) ? GetString(ind, "id") ?? string.Empty : string.Empty;
                string country = point.TryGetProperty("country", out var ctry) ? GetString(ctry, "id") ?? string.Empty : string.Empty;
                int year = int.TryParse(GetString(point, "date"), out int y) ? y : 0;
                double? value = point.TryGetProperty("value", out var v) && v.ValueKind == JsonValueKind.Number ? v.GetDouble() : null;
                points.Add(new WorldBankPoint(country, indicator, year, value));
            }
        }
        return points;
    }

    private static string? GetString(JsonElement parent, string name)
        => parent.ValueKind == JsonValueKind.Object && parent.TryGetProperty(name, out var element) &&
           element.ValueKind == JsonValueKind.String
            ? element.GetString()
            : null;
}
