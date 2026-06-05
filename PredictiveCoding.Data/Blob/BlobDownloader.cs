namespace PredictiveCoding.Data.Blob;

/// <summary>
/// Downloads blobs by URL — public or pre-signed object-storage URLs (Amazon S3,
/// Azure Blob, Google Cloud Storage) and plain HTTP(S). The downloaded bytes can
/// then be handed to the format-specific loaders (audio, video, images, …).
/// </summary>
public sealed class BlobDownloader : IDisposable
{
    private readonly HttpClient _http;
    private readonly bool _ownsHttp;

    public BlobDownloader(HttpClient? httpClient = null)
    {
        _http = httpClient ?? new HttpClient();
        _ownsHttp = httpClient is null;
    }

    /// <summary>Downloads a blob into memory.</summary>
    public async Task<byte[]> DownloadBytesAsync(string url, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(url);
        return await _http.GetByteArrayAsync(url, cancellationToken).ConfigureAwait(false);
    }

    /// <summary>Downloads a blob to a local file and returns its path.</summary>
    public async Task<string> DownloadToFileAsync(string url, string filePath, CancellationToken cancellationToken = default)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(url);
        ArgumentException.ThrowIfNullOrEmpty(filePath);

        using HttpResponseMessage response = await _http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken).ConfigureAwait(false);
        response.EnsureSuccessStatusCode();

        await using (var stream = File.Create(filePath))
            await response.Content.CopyToAsync(stream, cancellationToken).ConfigureAwait(false);
        return filePath;
    }

    public void Dispose()
    {
        if (_ownsHttp) _http.Dispose();
    }
}
