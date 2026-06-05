using System.Diagnostics;
using PredictiveCoding.Benchmark;
using PredictiveCoding.FaceRecognition;

// "gpu" sub-command: detect the CUDA stack and, with "gpu install", fetch the runtime headlessly.
if (args.Length > 0 && args[0].Equals("gpu", StringComparison.OrdinalIgnoreCase))
{
    string gpuRoot = Environment.GetEnvironmentVariable("FACEBENCH_DATA")
        ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "facebench-data");
    string cudaDir = Path.Combine(gpuRoot, ".cuda");
    GpuStatus s = CudaProvisioner.Detect(cudaDir);

    Console.WriteLine("== GPU / CUDA setup ==");
    Console.WriteLine($"   NVIDIA GPU          : {(s.HasNvidiaGpu ? $"{s.GpuName} (driver {s.DriverVersion})" : "not detected")}");
    Console.WriteLine($"   CUDA runtime present: {s.CudaRuntimePresent}");
    Console.WriteLine($"   cuDNN present       : {s.CudnnPresent}");
    Console.WriteLine($"   CUDA_PATH           : {(string.IsNullOrEmpty(s.CudaPath) ? "—" : s.CudaPath)}");

    bool install = args.Any(a => a.Equals("install", StringComparison.OrdinalIgnoreCase));
    if (install && s.HasNvidiaGpu && !(s.CudaRuntimePresent && s.CudnnPresent))
    {
        Console.WriteLine($"\n   downloading CUDA runtime + cuDNN redistributables (headless, no admin) → {cudaDir}");
        await CudaProvisioner.InstallAsync(cudaDir, m => Console.WriteLine("   " + m));
        Console.WriteLine($"\n   put it on PATH:  set \"PATH={cudaDir};%PATH%\"");
        Console.WriteLine("   then build the benchmark against Microsoft.ML.OnnxRuntime.Gpu to use the CUDA provider.");
    }
    else if (!install)
    {
        Console.WriteLine("\n   The benchmark runs ArcFace on the CUDA provider and auto-provisions these libraries on");
        Console.WriteLine("   first run. Use 'gpu install' to pre-fetch them now (headless, no admin).");
    }
    else
    {
        Console.WriteLine("\n   nothing to install (no NVIDIA GPU detected, or CUDA is already present).");
    }
    return 0;
}

// Data root: CLI arg, else FACEBENCH_DATA env var, else a folder in the user profile.
string dataRoot = args.Length > 0
    ? args[0]
    : Environment.GetEnvironmentVariable("FACEBENCH_DATA")
      ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), "facebench-data");
Directory.CreateDirectory(dataRoot);

Console.WriteLine("== PredictiveCoding.Benchmark — face-recognition benchmarks ==");
Console.WriteLine($"   data root : {dataRoot}");
Console.WriteLine("   embedder  : Predictive Coding (unsupervised latent, no backprop)");
Console.WriteLine("   index     : THEQULT VectorIndex (cosine) for 1:N identification\n");

using var http = new HttpClient { Timeout = TimeSpan.FromSeconds(300) };
http.DefaultRequestHeaders.UserAgent.ParseAdd("THEQULT-FaceBench/1.0"); // figshare needs a UA

int maxPairs = args.Length > 1 && int.TryParse(args[1], out int mp) ? mp : 6000;
string? embedderFilter = args.Length > 2 ? args[2] : null;
var budget = new BenchmarkBudget(MaxPairs: maxPairs);
int ran = 0, pending = 0;

// Hand-crafted LBP descriptor vs learned Predictive Coding latent vs a pretrained deep ArcFace model.
(string Label, Func<string, IFaceEmbedder> Factory)[] allEmbedders =
[
    ("LBP baseline", _ => new DescriptorEmbedder()),
    ("Predictive Coding", dir => new PcFaceEmbedder(dir)),
    ("ArcFace deep", _ => new ArcFaceOnnxEmbedder(Path.Combine(dataRoot, ".models"))),
];
(string Label, Func<string, IFaceEmbedder> Factory)[] embedders = embedderFilter is null
    ? allEmbedders
    : [.. allEmbedders.Where(e => e.Label.Contains(embedderFilter, StringComparison.OrdinalIgnoreCase))];

foreach (IFaceBenchmark bench in DatasetRegistry.All())
{
    Console.WriteLine($"── {bench.Name}  ({bench.Kind})");
    try
    {
        bool available = bench.IsAvailable(dataRoot);
        if (!available)
        {
            Console.Write("   non presente in locale — provo a scaricare… ");
            try { available = await bench.TryDownloadAsync(dataRoot, http); }
            catch (Exception ex) { Console.WriteLine($"fallito ({ex.Message})"); available = false; }
            if (available) Console.WriteLine("ok");
            else if (bench.Acquisition.Gated || bench.IsAvailable(dataRoot) == false) Console.WriteLine("non scaricabile");
        }

        if (!available)
        {
            pending++;
            Console.WriteLine($"   acquisire da: {bench.Acquisition.Url}");
            Console.WriteLine($"                 {bench.Acquisition.Notes}{(bench.Acquisition.Gated ? "   [GATED]" : "")}\n");
            continue;
        }

        foreach ((string label, Func<string, IFaceEmbedder> factory) in embedders)
        {
            string pcDir = Path.Combine(Path.GetTempPath(), "facebench-pc", bench.Name, Guid.NewGuid().ToString("N"));
            IFaceEmbedder embedder = factory(pcDir);
            try
            {
                var sw = Stopwatch.StartNew();
                string summary = bench.Run(dataRoot, embedder, budget);
                sw.Stop();
                Console.WriteLine($"   {summary}  ({sw.Elapsed.TotalSeconds:F1}s)");
            }
            finally
            {
                (embedder as IDisposable)?.Dispose();
                try { Directory.Delete(pcDir, recursive: true); } catch { /* best effort */ }
            }
        }
        Console.WriteLine();
        ran++;
    }
    catch (Exception ex)
    {
        Console.WriteLine($"   ERRORE: {ex.Message}\n");
    }
}

Console.WriteLine($"Riepilogo: {ran} dataset eseguiti, {pending} in attesa dei dati.");
return 0;
