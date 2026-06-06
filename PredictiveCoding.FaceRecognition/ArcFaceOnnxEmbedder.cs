using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PredictiveCoding.FaceRecognition.FaceAlignment;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PredictiveCoding.FaceRecognition;

/// <summary>
/// The "how the best do it" embedder: a pretrained <b>ArcFace</b> deep model (InsightFace <c>w600k_r50</c>,
/// trained on Glint360K with an additive-angular-margin loss) run via ONNX Runtime. Produces the standard
/// 512-D face embedding; cosine similarity then drives verification / identification.
/// <para>
/// When <paramref name="useAlignment"/> is <see langword="true"/> (the default), the
/// <see cref="RetinaFaceDetector"/> detects the face and computes 5-point landmarks;
/// <see cref="FaceAligner"/> then warps the image to the canonical 112×112 ArcFace template via
/// a closed-form similarity transform — the same pipeline used by InsightFace in production.
/// If no face is detected, it falls back to a centre-crop.
/// </para>
/// </summary>
public sealed class ArcFaceOnnxEmbedder : IFaceEmbedder, IDisposable
{
    // InsightFace buffalo_l recognition model, public mirror (no auth).
    private const string ModelUrl = "https://huggingface.co/immich-app/buffalo_l/resolve/main/recognition/model.onnx";

    private readonly string  _modelsDir;
    private readonly string  _cudaDir;
    private readonly float   _cropFraction;
    private readonly bool    _useAlignment;
    private const    int     BatchSize = 64;
    private InferenceSession?  _session;
    private string?            _inputName;
    private string?            _provider;
    private int                _dimension = 512;
    private bool               _supportsBatch = true;
    private RetinaFaceDetector? _detector;    // null until EnsureLoaded() when _useAlignment=true

    public ArcFaceOnnxEmbedder(string modelsDir, float cropFraction = 0.50f, bool useAlignment = true)
    {
        _modelsDir    = modelsDir;
        _cudaDir      = Path.Combine(Path.GetDirectoryName(Path.GetFullPath(modelsDir)) ?? modelsDir, ".cuda");
        _cropFraction = cropFraction;
        _useAlignment = useAlignment;
    }

    public int    Dimension => _dimension;
    public string Name      => $"ArcFace w600k_r50{(_useAlignment ? " +align" : "")} · {_provider ?? "ONNX"}";

    public void Fit(IReadOnlyList<string> trainingImagePaths) => EnsureLoaded(); // pretrained: nothing to learn

    public float[] Embed(string imagePath)
    {
        EnsureLoaded();
        var input = new DenseTensor<float>([1, 3, 112, 112]);
        if (!PreprocessInto(imagePath, input, 0))
            return new float[_dimension];

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results =
            _session!.Run([NamedOnnxValue.CreateFromTensor(_inputName!, input)]);
        float[] embedding = results[0].AsEnumerable<float>().ToArray();
        Normalize(embedding);
        return embedding;
    }

    /// <summary>Runs the GPU in real batches: decode/preprocess in parallel, one <c>Run</c> per chunk.</summary>
    public IReadOnlyList<float[]?> EmbedBatch(IReadOnlyList<string> imagePaths)
    {
        EnsureLoaded();
        var result = new float[]?[imagePaths.Count];

        for (int start = 0; start < imagePaths.Count && _supportsBatch; start += BatchSize)
        {
            int n = Math.Min(BatchSize, imagePaths.Count - start);
            var input = new DenseTensor<float>([n, 3, 112, 112]);
            var ok = new bool[n];
            Parallel.For(0, n, k => ok[k] = PreprocessInto(imagePaths[start + k], input, k));

            float[] flat;
            try
            {
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results =
                    _session!.Run([NamedOnnxValue.CreateFromTensor(_inputName!, input)]);
                flat = results[0].AsEnumerable<float>().ToArray();
            }
            catch
            {
                _supportsBatch = false; // fixed-batch model: fall back to per-image below
                break;
            }

            for (int k = 0; k < n; k++)
            {
                if (!ok[k])
                    continue;
                var v = new float[_dimension];
                Array.Copy(flat, k * _dimension, v, 0, _dimension);
                Normalize(v);
                result[start + k] = v;
            }
        }

        // Fallback path (DirectML, or a chunk that fell back): the DML device is not safe for concurrent Run
        // and rejects batched BatchNorm, so decode in parallel (CPU-bound) but run sequentially, chunked to
        // bound memory.
        const int decodeChunk = 256;
        for (int start = 0; start < result.Length; start += decodeChunk)
        {
            int n = Math.Min(decodeChunk, result.Length - start);
            var prepared = new DenseTensor<float>?[n];
            Parallel.For(0, n, k =>
            {
                if (result[start + k] is not null)
                    return;
                var tensor = new DenseTensor<float>([1, 3, 112, 112]);
                prepared[k] = PreprocessInto(imagePaths[start + k], tensor, 0) ? tensor : null;
            });

            for (int k = 0; k < n; k++)
            {
                if (result[start + k] is not null || prepared[k] is null)
                    continue;
                using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> res =
                    _session!.Run([NamedOnnxValue.CreateFromTensor(_inputName!, prepared[k]!)]);
                float[] v = res[0].AsEnumerable<float>().ToArray();
                Normalize(v);
                result[start + k] = v;
            }
        }
        return result;
    }

    private bool PreprocessInto(string imagePath, DenseTensor<float> tensor, int batchIndex)
    {
        try
        {
            using Image<Rgb24> image = Image.Load<Rgb24>(imagePath);

            // Preferred path: detect face → 5-point similarity-transform alignment → 112×112
            Image<Rgb24>? aligned = null;
            try
            {
                if (_detector is not null)
                {
                    FaceDetection? det = _detector.Detect(image);
                    if (det is not null)
                        aligned = FaceAligner.AlignFace(image, det.Landmarks);
                }
            }
            catch { /* detection failure: fall through to centre-crop */ }

            // Fallback: centre-crop (original behaviour)
            if (aligned is null)
            {
                int side = Math.Max(1, (int)(Math.Min(image.Width, image.Height) * _cropFraction));
                int x0   = (image.Width  - side) / 2;
                int y0   = (image.Height - side) / 2;
                image.Mutate(ctx => ctx.Crop(new Rectangle(x0, y0, side, side)).Resize(112, 112));
                aligned = image;
            }

            // Write NCHW float tensor: (pixel - 127.5) / 127.5  →  [−1, +1]
            bool usedAlignedCopy = !ReferenceEquals(aligned, image);
            try
            {
                aligned.ProcessPixelRows(accessor =>
                {
                    for (int y = 0; y < 112; y++)
                    {
                        Span<Rgb24> row = accessor.GetRowSpan(y);
                        for (int x = 0; x < 112; x++)
                        {
                            Rgb24 p = row[x];
                            tensor[batchIndex, 0, y, x] = (p.R - 127.5f) / 127.5f;
                            tensor[batchIndex, 1, y, x] = (p.G - 127.5f) / 127.5f;
                            tensor[batchIndex, 2, y, x] = (p.B - 127.5f) / 127.5f;
                        }
                    }
                });
            }
            finally
            {
                if (usedAlignedCopy) aligned.Dispose();
            }
            return true;
        }
        catch
        {
            return false;
        }
    }

    private static void Normalize(float[] v)
    {
        double norm = 0;
        foreach (float x in v)
            norm += (double)x * x;
        norm = Math.Sqrt(norm);
        if (norm > 0)
            for (int i = 0; i < v.Length; i++)
                v[i] = (float)(v[i] / norm);
    }

    private void EnsureLoaded()
    {
        if (_session is not null)
            return;

        Directory.CreateDirectory(_modelsDir);
        string modelPath = Path.Combine(_modelsDir, "arcface_w600k_r50.onnx");
        if (!File.Exists(modelPath) || new FileInfo(modelPath).Length < 1_000_000)
            Download(ModelUrl, modelPath);

        EnsureCudaRuntime();
        _session = CreateSession(modelPath, out _provider);
        _inputName = _session.InputMetadata.Keys.First();
        int[] outDims = _session.OutputMetadata.First().Value.Dimensions;
        _dimension = outDims.Length > 0 && outDims[^1] > 0 ? outDims[^1] : 512;
        // Real batches only when the model truly has a dynamic batch axis. This one is exported batch=1
        // (output {1,512}); feeding N>1 hits a slow dynamic path, so use parallel-decode + sequential runs.
        _supportsBatch = outDims.Length > 0 && outDims[0] != 1;
        Console.WriteLine($"   [ArcFace] execution provider: {_provider}  (batch: {(_supportsBatch ? "on" : "parallel per-image")})");

        // Optionally initialise the RetinaFace 5-point alignment detector (lazy model download)
        if (_useAlignment)
        {
            _detector = new RetinaFaceDetector(_modelsDir);
            Console.WriteLine("   [ArcFace] face alignment: RetinaFace det_10g (5-point, downloads on first use)");
        }
    }

    // Headless, no-admin self-provisioning: ensure the CUDA runtime + cuDNN DLLs exist and are on the process
    // PATH so ONNX Runtime's CUDA provider can load — nothing for the user to install or configure.
    private void EnsureCudaRuntime()
    {
        try
        {
            GpuStatus status = CudaProvisioner.Detect(_cudaDir);
            if (status.HasNvidiaGpu && (!status.CudaRuntimePresent || !status.CudnnPresent))
            {
                Console.WriteLine("   [ArcFace] provisioning CUDA runtime + cuDNN (first run, headless, no admin)…");
                CudaProvisioner.InstallAsync(_cudaDir, m => Console.WriteLine("   [cuda] " + m)).GetAwaiter().GetResult();
            }
            if (Directory.Exists(_cudaDir))
            {
                string path = Environment.GetEnvironmentVariable("PATH") ?? "";
                if (!path.Split(Path.PathSeparator).Contains(_cudaDir, StringComparer.OrdinalIgnoreCase))
                    Environment.SetEnvironmentVariable("PATH", _cudaDir + Path.PathSeparator + path);
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("   [ArcFace] CUDA runtime setup skipped: " + ex.Message);
        }
    }

    private static InferenceSession CreateSession(string modelPath, out string provider)
    {
        try
        {
            // ERROR severity silences the benign "{1,512} vs {64,512}" batch shape warning this model emits.
            var options = new SessionOptions { LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };
            // Heuristic cuDNN conv selection avoids the slow one-time exhaustive search at first inference.
            var cudaOptions = new OrtCUDAProviderOptions();
            cudaOptions.UpdateOptions(new Dictionary<string, string> { ["cudnn_conv_algo_search"] = "HEURISTIC" });
            options.AppendExecutionProvider_CUDA(cudaOptions); // NVIDIA CUDA EP (uses the runtime DLLs put on PATH above)
            cudaOptions.Dispose();
            var session = new InferenceSession(modelPath, options);
            provider = "CUDA (GPU)";
            return session;
        }
        catch (Exception ex)
        {
            provider = $"CPU (CUDA unavailable: {ex.Message.Split('\n', '\r')[0]})";
            return new InferenceSession(modelPath);
        }
    }

    private static void Download(string url, string destination)
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("THEQULT-FaceBench/1.0");
        using HttpResponseMessage resp = http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
        resp.EnsureSuccessStatusCode();
        string tmp = destination + ".part";
        using (FileStream fs = File.Create(tmp))
            resp.Content.CopyToAsync(fs).GetAwaiter().GetResult();
        File.Move(tmp, destination, overwrite: true);
    }

    public void Dispose()
    {
        _session?.Dispose();
        _detector?.Dispose();
    }
}
