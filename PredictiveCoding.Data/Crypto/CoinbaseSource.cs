using System.Globalization;
using System.Text.Json;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Crypto;

/// <summary>
/// Cryptocurrency data source backed by the public Coinbase APIs: spot price and
/// historic candles. JSON parsing is static and runs offline.
/// </summary>
public sealed class CoinbaseSource : IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public CoinbaseSource(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Fetches the current spot price for a pair such as <c>BTC-USD</c>.</summary>
    public async Task<decimal> GetSpotPriceAsync(string pair = "BTC-USD", CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(pair);
        string url = $"https://api.coinbase.com/v2/prices/{Uri.EscapeDataString(pair)}/spot";
        return ParseSpotPrice(await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    /// <summary>
    /// Fetches historic candles for a product (Coinbase Exchange API), returned as a
    /// <c>[count, 6]</c> tensor of [time, low, high, open, close, volume].
    /// </summary>
    public async Task<DenseTensor<float>> GetCandlesAsync(
        string productId = "BTC-USD", int granularitySeconds = 3600, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(productId);
        string url = $"https://api.exchange.coinbase.com/products/{Uri.EscapeDataString(productId)}/candles"
                   + $"?granularity={granularitySeconds}";
        return ParseCandles(await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses a Coinbase spot-price response (offline).</summary>
    public static decimal ParseSpotPrice(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        if (document.RootElement.TryGetProperty("data", out var data) &&
            data.TryGetProperty("amount", out var amount) && amount.ValueKind == JsonValueKind.String &&
            decimal.TryParse(amount.GetString(), NumberStyles.Number, CultureInfo.InvariantCulture, out decimal value))
            return value;
        throw new FormatException("Coinbase spot-price response did not contain a numeric amount.");
    }

    /// <summary>Parses a Coinbase candles response into a <c>[count, 6]</c> tensor (offline).</summary>
    public static DenseTensor<float> ParseCandles(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        if (document.RootElement.ValueKind != JsonValueKind.Array)
            throw new FormatException("Coinbase candles response is not an array.");

        var rows = new List<float[]>();
        foreach (JsonElement candle in document.RootElement.EnumerateArray())
        {
            if (candle.ValueKind != JsonValueKind.Array)
                continue;
            var row = new float[6];
            int i = 0;
            foreach (JsonElement field in candle.EnumerateArray())
            {
                if (i >= 6) break;
                row[i++] = field.ValueKind == JsonValueKind.Number ? (float)field.GetDouble() : float.NaN;
            }
            rows.Add(row);
        }

        if (rows.Count == 0)
            throw new FormatException("Coinbase candles response contained no candles.");

        var data = new float[rows.Count * 6];
        for (int r = 0; r < rows.Count; r++)
            Array.Copy(rows[r], 0, data, r * 6, 6);
        return new DenseTensor<float>(data, rows.Count, 6);
    }
}
