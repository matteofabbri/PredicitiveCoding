using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json;

namespace PredictiveCoding.FaceRecognition;

/// <summary>What we found about the machine's GPU / CUDA stack.</summary>
public sealed record GpuStatus(
    bool HasNvidiaGpu, string? GpuName, string? DriverVersion,
    bool CudaRuntimePresent, bool CudnnPresent, string? CudaPath);

/// <summary>
/// Detects the GPU + CUDA stack and, when an NVIDIA GPU is present but the CUDA runtime / cuDNN are missing,
/// installs them <b>headlessly and without admin</b> by fetching NVIDIA's official <i>redistributable</i> DLL
/// archives (the runtime libraries only — no multi-GB Toolkit, no installer, no UAC). The DLLs are dropped into
/// a folder you can put on <c>PATH</c> so ONNX Runtime's CUDA execution provider can load.
/// </summary>
public static class CudaProvisioner
{
    private const string CudaBase = "https://developer.download.nvidia.com/compute/cuda/redist/";
    private const string CudaManifest = "redistrib_12.6.3.json";   // CUDA 12.6 runtime
    private const string CudnnBase = "https://developer.download.nvidia.com/compute/cudnn/redist/";
    private const string CudnnManifest = "redistrib_9.5.1.json";   // cuDNN 9.5 for CUDA 12

    // The runtime libraries ONNX Runtime's CUDA EP depends on.
    private static readonly string[] CudaComponents =
        ["cuda_cudart", "libcublas", "libcufft", "libcurand", "libcusparse", "libnvjitlink"];

    public static GpuStatus Detect(string? extraSearchDir = null)
    {
        (bool hasGpu, string? name, string? driver) = QueryNvidiaSmi();
        string? cudaPath = Environment.GetEnvironmentVariable("CUDA_PATH");
        bool cudart = FindDll("cudart64_*.dll", extraSearchDir, cudaPath) is not null;
        bool cudnn = FindDll("cudnn64_*.dll", extraSearchDir, cudaPath) is not null;
        return new GpuStatus(hasGpu, name, driver, cudart, cudnn, cudaPath);
    }

    /// <summary>Downloads and extracts the CUDA runtime + cuDNN redistributable DLLs into <paramref name="targetDir"/>.</summary>
    public static async Task InstallAsync(string targetDir, Action<string> log, bool full = true)
    {
        Directory.CreateDirectory(targetDir);
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(30) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("THEQULT-FaceBench/1.0");

        log($"fetching CUDA manifest {CudaManifest}…");
        using JsonDocument cuda = JsonDocument.Parse(await http.GetStringAsync(CudaBase + CudaManifest));
        foreach (string comp in full ? CudaComponents : ["cuda_cudart"])
        {
            if (cuda.RootElement.TryGetProperty(comp, out JsonElement c) &&
                c.TryGetProperty("windows-x86_64", out JsonElement w) &&
                w.TryGetProperty("relative_path", out JsonElement rel))
            {
                await DownloadDllsAsync(http, CudaBase + rel.GetString(), targetDir, log);
            }
        }

        if (full)
        {
            log($"fetching cuDNN manifest {CudnnManifest}…");
            using JsonDocument cudnn = JsonDocument.Parse(await http.GetStringAsync(CudnnBase + CudnnManifest));
            if (cudnn.RootElement.TryGetProperty("cudnn", out JsonElement cu) &&
                cu.TryGetProperty("windows-x86_64", out JsonElement cw) &&
                cw.TryGetProperty("cuda12", out JsonElement c12) &&
                c12.TryGetProperty("relative_path", out JsonElement crel))
            {
                await DownloadDllsAsync(http, CudnnBase + crel.GetString(), targetDir, log);
            }
        }

        log($"done → {targetDir}");
    }

    private static async Task DownloadDllsAsync(HttpClient http, string url, string targetDir, Action<string> log)
    {
        log($"  ↓ {Path.GetFileName(url)}");
        byte[] zip = await http.GetByteArrayAsync(url);
        using var archive = new ZipArchive(new MemoryStream(zip));
        int extracted = 0;
        foreach (ZipArchiveEntry entry in archive.Entries)
        {
            if (!entry.FullName.EndsWith(".dll", StringComparison.OrdinalIgnoreCase))
                continue;
            if (!entry.FullName.Replace('\\', '/').Contains("/bin/", StringComparison.OrdinalIgnoreCase))
                continue;
            entry.ExtractToFile(Path.Combine(targetDir, Path.GetFileName(entry.FullName)), overwrite: true);
            extracted++;
        }
        log($"    extracted {extracted} DLL(s)  ({zip.Length / (1024 * 1024)} MB)");
    }

    private static (bool, string?, string?) QueryNvidiaSmi()
    {
        try
        {
            var psi = new ProcessStartInfo("nvidia-smi", "--query-gpu=name,driver_version --format=csv,noheader,nounits")
            {
                RedirectStandardOutput = true,
                UseShellExecute = false,
                CreateNoWindow = true,
            };
            using Process? p = Process.Start(psi);
            if (p is null)
                return (false, null, null);
            string output = p.StandardOutput.ReadToEnd();
            p.WaitForExit(5000);
            string first = output.Split('\n', StringSplitOptions.RemoveEmptyEntries).FirstOrDefault()?.Trim() ?? "";
            if (first.Length == 0)
                return (false, null, null);
            string[] parts = first.Split(',');
            return (true, parts[0].Trim(), parts.Length > 1 ? parts[1].Trim() : null);
        }
        catch
        {
            return (false, null, null);
        }
    }

    private static string? FindDll(string pattern, params string?[] extraDirs)
    {
        var dirs = new List<string>();
        dirs.AddRange((Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator));
        foreach (string? d in extraDirs)
        {
            if (string.IsNullOrEmpty(d))
                continue;
            dirs.Add(d);
            dirs.Add(Path.Combine(d, "bin"));
        }

        foreach (string dir in dirs)
        {
            if (string.IsNullOrWhiteSpace(dir) || !Directory.Exists(dir))
                continue;
            try
            {
                string? hit = Directory.EnumerateFiles(dir, pattern).FirstOrDefault();
                if (hit is not null)
                    return hit;
            }
            catch
            {
                // unreadable PATH entry — skip
            }
        }
        return null;
    }
}
