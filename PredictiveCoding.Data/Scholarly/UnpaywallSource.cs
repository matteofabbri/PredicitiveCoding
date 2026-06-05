using System.Text.Json;

namespace PredictiveCoding.Data.Scholarly;

/// <summary>An Unpaywall record: a paper's legal open-access locations (if any).</summary>
public sealed record UnpaywallRecord(
    string Doi,
    string? Title,
    bool IsOpenAccess,
    string? BestPdfUrl,
    IReadOnlyList<string> PdfUrls);

/// <summary>
/// Finds legal open-access copies of papers by DOI via Unpaywall (the lawful way
/// to obtain free full text — it indexes only legitimately deposited OA copies).
/// JSON parsing is static and runs offline. Unpaywall requires a contact email.
/// </summary>
public sealed class UnpaywallSource : IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;
    private readonly string _email;

    public UnpaywallSource(string email, HttpClient? httpClient = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(email);
        _email = email;
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Looks up the open-access status and locations for a DOI.</summary>
    public async Task<UnpaywallRecord> GetAsync(string doi, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(doi);
        string url = $"https://api.unpaywall.org/v2/{Uri.EscapeDataString(doi)}?email={Uri.EscapeDataString(_email)}";
        return Parse(await _http.GetStringAsync(url, cancellationToken).ConfigureAwait(false));
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }

    /// <summary>Parses an Unpaywall response (offline).</summary>
    public static UnpaywallRecord Parse(string json)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(json);
        using var document = JsonDocument.Parse(json);
        JsonElement root = document.RootElement;

        string doi = GetString(root, "doi") ?? string.Empty;
        string? title = GetString(root, "title");
        bool isOa = root.TryGetProperty("is_oa", out var oa) && oa.ValueKind == JsonValueKind.True;

        string? bestPdf = null;
        if (root.TryGetProperty("best_oa_location", out var best) && best.ValueKind == JsonValueKind.Object)
            bestPdf = GetString(best, "url_for_pdf") ?? GetString(best, "url");

        var pdfUrls = new List<string>();
        if (root.TryGetProperty("oa_locations", out var locations) && locations.ValueKind == JsonValueKind.Array)
        {
            foreach (JsonElement location in locations.EnumerateArray())
            {
                string? pdf = GetString(location, "url_for_pdf") ?? GetString(location, "url");
                if (!string.IsNullOrWhiteSpace(pdf))
                    pdfUrls.Add(pdf);
            }
        }

        return new UnpaywallRecord(doi, title, isOa, bestPdf, pdfUrls);
    }

    private static string? GetString(JsonElement parent, string name)
        => parent.ValueKind == JsonValueKind.Object && parent.TryGetProperty(name, out var element) &&
           element.ValueKind == JsonValueKind.String
            ? element.GetString()
            : null;
}
