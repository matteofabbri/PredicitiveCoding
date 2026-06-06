using System.Formats.Tar;
using System.IO.Compression;
using System.Net.Http.Headers;

namespace PredictiveCoding.FaceRecognition.Datasets;

/// <summary>
/// Downloads publicly available face recognition datasets.
/// LFW is downloaded directly (~233 MB); large-scale training sets are fetched from HuggingFace Hub.
/// </summary>
public static class FaceDatasetDownloader
{
    // ── LFW endpoints ─────────────────────────────────────────────────────────────
    private const string LfwDeepFunneledUrl = "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz";
    private const string LfwUrl             = "http://vis-www.cs.umass.edu/lfw/lfw.tgz";
    private const string LfwPairsUrl        = "http://vis-www.cs.umass.edu/lfw/pairs.txt";

    // ── Well-known HuggingFace datasets ──────────────────────────────────────────
    public static readonly DatasetInfo[] LargeScaleDatasets =
    [
        new("vggface2-dataset/vggface2",      "VGGFace2",        "~36 GB, 3.31 M images, 9131 subjects — GATED, requires sign-up"),
        new("CASIA/CASIA-WebFace",            "CASIA-WebFace",   "~4 GB, 490K images, 10575 subjects — GATED"),
        new("nateraw/celebahq-256",           "CelebA-HQ 256",   "~7 GB, 30K high-res images — PUBLIC (no token needed)"),
        new("deepinsight/ms1m-retinaface-t1", "MS1M-RetinaFace", "~87 GB, 5.1 M images, 93K subjects — GATED"),
    ];

    // ── LFW ──────────────────────────────────────────────────────────────────────

    /// <summary>
    /// Downloads and extracts LFW to <paramref name="dataDir"/>.
    /// Returns the path to the images root (one sub-folder per identity).
    /// </summary>
    public static async Task<string> DownloadLfwAsync(
        string          dataDir,
        bool            deepFunneled = true,
        Action<string>? onProgress   = null,
        CancellationToken cancellation = default)
    {
        Directory.CreateDirectory(dataDir);

        string archiveName     = deepFunneled ? "lfw-deepfunneled.tgz" : "lfw.tgz";
        string extractedFolder = deepFunneled ? "lfw-deepfunneled" : "lfw";
        string downloadUrl     = deepFunneled ? LfwDeepFunneledUrl : LfwUrl;
        string archivePath     = Path.Combine(dataDir, archiveName);
        string imagesRoot      = Path.Combine(dataDir, extractedFolder);

        // Pairs file
        string pairsPath = Path.Combine(dataDir, "pairs.txt");
        if (!File.Exists(pairsPath))
        {
            onProgress?.Invoke("Downloading pairs.txt …");
            await DownloadAsync(LfwPairsUrl, pairsPath, null, cancellation);
        }

        // Images
        bool alreadyExtracted = Directory.Exists(imagesRoot) &&
                                Directory.EnumerateDirectories(imagesRoot).Any();
        if (!alreadyExtracted)
        {
            if (!IsValidArchive(archivePath))
            {
                string variant = deepFunneled ? "deep-funneled" : "original";
                onProgress?.Invoke($"Downloading LFW {variant} (~233 MB) …");
                await DownloadAsync(downloadUrl, archivePath,
                    (got, total) =>
                    {
                        if (total > 0)
                            onProgress?.Invoke(
                                $"  {got / 1_048_576:F0} / {total / 1_048_576:F0} MB  ({100.0 * got / total:F0}%)");
                    },
                    cancellation);
            }

            onProgress?.Invoke($"Extracting {archiveName} …");
            await ExtractTgzAsync(archivePath, dataDir, cancellation);
            onProgress?.Invoke("Extraction complete.");
        }
        else
        {
            onProgress?.Invoke($"LFW already present at {imagesRoot}");
        }

        return imagesRoot;
    }

    // ── HuggingFace Hub ───────────────────────────────────────────────────────────

    /// <summary>
    /// Downloads a single file from a HuggingFace Hub repository.
    /// For gated datasets, set <paramref name="hfToken"/> or the HF_TOKEN environment variable.
    /// </summary>
    public static async Task<string> DownloadHuggingFaceFileAsync(
        string          dataDir,
        string          repoId,
        string          repoPath,
        bool            isDataset = true,
        string?         hfToken   = null,
        Action<string>? onProgress = null,
        CancellationToken cancellation = default)
    {
        Directory.CreateDirectory(dataDir);
        string token = hfToken ?? Environment.GetEnvironmentVariable("HF_TOKEN") ?? "";
        string repoType = isDataset ? "datasets" : "models";
        string url = $"https://huggingface.co/{repoType}/{repoId}/resolve/main/{repoPath}";
        string localPath = Path.Combine(dataDir, Path.GetFileName(repoPath));

        onProgress?.Invoke($"HuggingFace ▸ {repoId}/{repoPath}");
        if (string.IsNullOrEmpty(token))
            onProgress?.Invoke("  (no HF_TOKEN — will fail for gated datasets)");

        using HttpClient http = BuildHttp(token, TimeSpan.FromHours(3));
        await DownloadAsync(url, localPath,
            (got, total) =>
            {
                if (total > 0)
                    onProgress?.Invoke(
                        $"  {got / 1_048_576:F0} / {total / 1_048_576:F0} MB  ({100.0 * got / total:F0}%)");
            },
            cancellation, http);

        return localPath;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────────

    private static async Task DownloadAsync(
        string            url,
        string            destPath,
        Action<long,long>? onProgress,
        CancellationToken  cancellation,
        HttpClient?        http = null)
    {
        bool owns = http is null;
        http ??= BuildHttp("", TimeSpan.FromHours(2));
        string tmp = destPath + ".part";
        try
        {
            using HttpResponseMessage resp = await http.GetAsync(
                url, HttpCompletionOption.ResponseHeadersRead, cancellation);
            resp.EnsureSuccessStatusCode();

            long total = resp.Content.Headers.ContentLength ?? -1;
            long got   = 0;

            await using FileStream fs  = File.Create(tmp);
            await using Stream     src = await resp.Content.ReadAsStreamAsync(cancellation);
            byte[] buf = new byte[131_072];
            int read;
            while ((read = await src.ReadAsync(buf, cancellation)) > 0)
            {
                await fs.WriteAsync(buf.AsMemory(0, read), cancellation);
                got += read;
                onProgress?.Invoke(got, total);
            }
        }
        finally
        {
            if (owns) http.Dispose();
        }
        File.Move(tmp, destPath, overwrite: true);
    }

    private static async Task ExtractTgzAsync(
        string tgzPath, string destDir, CancellationToken cancellation)
    {
        await using FileStream fs = File.OpenRead(tgzPath);
        await using GZipStream gz = new(fs, CompressionMode.Decompress);
        await TarFile.ExtractToDirectoryAsync(gz, destDir, overwriteFiles: true, cancellation);
    }

    private static bool IsValidArchive(string path) =>
        File.Exists(path) && new FileInfo(path).Length > 10_000_000;

    private static HttpClient BuildHttp(string token, TimeSpan timeout)
    {
        var client = new HttpClient { Timeout = timeout };
        client.DefaultRequestHeaders.UserAgent.ParseAdd("PredictiveCoding-Face/1.0");
        if (!string.IsNullOrEmpty(token))
            client.DefaultRequestHeaders.Authorization =
                new AuthenticationHeaderValue("Bearer", token);
        return client;
    }
}

public record DatasetInfo(string RepoId, string Name, string Description);
