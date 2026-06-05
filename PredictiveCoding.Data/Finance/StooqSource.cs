using System.Globalization;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Finance;

/// <summary>A daily OHLCV price series.</summary>
public sealed class OhlcvSeries
{
    /// <summary>The column order of <see cref="Values"/>.</summary>
    public static IReadOnlyList<string> Columns { get; } = ["Open", "High", "Low", "Close", "Volume"];

    /// <summary>One date per row.</summary>
    public IReadOnlyList<DateOnly> Dates { get; }

    /// <summary>Values, shape <c>[rows, 5]</c> (open, high, low, close, volume).</summary>
    public DenseTensor<float> Values { get; }

    public OhlcvSeries(IReadOnlyList<DateOnly> dates, DenseTensor<float> values)
    {
        ArgumentNullException.ThrowIfNull(dates);
        ArgumentNullException.ThrowIfNull(values);
        if (values.Rank != 2 || values.Dimensions[0] != dates.Count || values.Dimensions[1] != 5)
            throw new ArgumentException("Values shape must be [dates, 5].", nameof(values));
        Dates = dates;
        Values = values;
    }
}

/// <summary>
/// Financial OHLCV source backed by Stooq (which returns CSV). CSV parsing is
/// static and runs offline.
/// </summary>
public sealed class StooqSource : IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public StooqSource(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Fetches the daily history of a symbol (e.g. <c>aapl.us</c>).</summary>
    public async Task<OhlcvSeries> GetDailyAsync(string symbol, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(symbol);
        string url = $"https://stooq.com/q/d/l/?s={Uri.EscapeDataString(symbol)}&i=d";
        return ParseCsv(await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses a Stooq daily CSV (<c>Date,Open,High,Low,Close,Volume</c>) offline.</summary>
    public static OhlcvSeries ParseCsv(string csv)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(csv);
        string[] lines = csv.Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        if (lines.Length < 2)
            throw new InvalidDataException("Stooq CSV contains no data rows.");

        var dates = new List<DateOnly>();
        var rows = new List<float[]>();
        for (int i = 1; i < lines.Length; i++) // skip header
        {
            string[] fields = lines[i].Split(',');
            if (fields.Length < 6)
                continue;
            if (!DateOnly.TryParse(fields[0], CultureInfo.InvariantCulture, DateTimeStyles.None, out DateOnly date))
                continue;

            var row = new float[5];
            for (int c = 0; c < 5; c++)
                float.TryParse(fields[c + 1], NumberStyles.Float, CultureInfo.InvariantCulture, out row[c]);
            dates.Add(date);
            rows.Add(row);
        }

        if (rows.Count == 0)
            throw new InvalidDataException("Stooq CSV contained no parseable rows.");

        var data = new float[rows.Count * 5];
        for (int r = 0; r < rows.Count; r++)
            Array.Copy(rows[r], 0, data, r * 5, 5);
        return new OhlcvSeries(dates, new DenseTensor<float>(data, rows.Count, 5));
    }
}
