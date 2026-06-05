using System.Globalization;
using System.Text.Json;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Weather;

/// <summary>An hourly weather time series: timestamps and a value matrix.</summary>
public sealed class WeatherSeries
{
    /// <summary>One timestamp per row.</summary>
    public IReadOnlyList<DateTimeOffset> Times { get; }

    /// <summary>The variable names, one per column.</summary>
    public IReadOnlyList<string> Variables { get; }

    /// <summary>Values, shape <c>[steps, variables]</c>.</summary>
    public DenseTensor<float> Values { get; }

    public WeatherSeries(IReadOnlyList<DateTimeOffset> times, IReadOnlyList<string> variables, DenseTensor<float> values)
    {
        ArgumentNullException.ThrowIfNull(times);
        ArgumentNullException.ThrowIfNull(variables);
        ArgumentNullException.ThrowIfNull(values);
        if (values.Rank != 2 || values.Dimensions[0] != times.Count || values.Dimensions[1] != variables.Count)
            throw new ArgumentException("Values shape must be [times, variables].", nameof(values));
        Times = times;
        Variables = variables;
        Values = values;
    }
}

/// <summary>
/// Weather data source backed by the free Open-Meteo API. JSON parsing is static
/// and runs offline.
/// </summary>
public sealed class WeatherSource : IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public WeatherSource(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Fetches an hourly forecast for the given location and variables.</summary>
    public async Task<WeatherSeries> GetHourlyAsync(
        double latitude, double longitude, IReadOnlyList<string> variables, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(variables);
        if (variables.Count == 0) throw new ArgumentException("At least one variable is required.", nameof(variables));

        string vars = string.Join(',', variables);
        string url = $"https://api.open-meteo.com/v1/forecast"
                   + $"?latitude={latitude.ToString(CultureInfo.InvariantCulture)}"
                   + $"&longitude={longitude.ToString(CultureInfo.InvariantCulture)}"
                   + $"&hourly={Uri.EscapeDataString(vars)}";
        string json = await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false);
        return ParseHourly(json);
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses an Open-Meteo hourly response (offline).</summary>
    public static WeatherSeries ParseHourly(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        if (!document.RootElement.TryGetProperty("hourly", out JsonElement hourly) || hourly.ValueKind != JsonValueKind.Object)
            throw new FormatException("Open-Meteo response is missing an 'hourly' object.");

        var times = new List<DateTimeOffset>();
        if (hourly.TryGetProperty("time", out JsonElement timeArray) && timeArray.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement t in timeArray.EnumerateArray())
            {
                if (DateTimeOffset.TryParse(t.GetString(), CultureInfo.InvariantCulture,
                        DateTimeStyles.AssumeUniversal, out var dto))
                    times.Add(dto);
            }
        }

        var variables = new List<string>();
        var columns = new List<float[]>();
        foreach (JsonProperty property in hourly.EnumerateObject())
        {
            if (property.Name == "time" || property.Value.ValueKind != JsonValueKind.Array)
                continue;
            variables.Add(property.Name);
            var values = new float[times.Count];
            int i = 0;
            foreach (JsonElement v in property.Value.EnumerateArray())
            {
                if (i >= values.Length) break;
                values[i++] = v.ValueKind == JsonValueKind.Number ? (float)v.GetDouble() : float.NaN;
            }
            columns.Add(values);
        }

        if (times.Count == 0 || variables.Count == 0)
            throw new FormatException("Open-Meteo response contained no hourly data.");

        var data = new float[times.Count * variables.Count];
        for (int r = 0; r < times.Count; r++)
            for (int c = 0; c < variables.Count; c++)
                data[r * variables.Count + c] = columns[c][r];

        return new WeatherSeries(times, variables, new DenseTensor<float>(data, times.Count, variables.Count));
    }
}
