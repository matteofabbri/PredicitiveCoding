using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PredictiveCoding.FaceRecognition;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PredictiveCoding.Vision.Detectors;

/// <summary>
/// Full-body person detector backed by the OpenCV-Zoo <b>YOLOX</b> model
/// (<c>object_detection_yolox_2022nov.onnx</c>, COCO-trained, Apache-2.0). It
/// returns every "person" (COCO class 0) box in a frame after NMS — the core of
/// per-frame people counting. The model (~36 MB) is downloaded once on first use.
/// <para>
/// Follows the same shape as <c>RetinaFaceDetector</c>: download → NCHW tensor →
/// <see cref="InferenceSession"/> → decode → NMS. YOLOX takes <b>BGR</b> pixels in
/// raw 0–255 (no mean/std normalisation) on a 640×640 letterboxed canvas.
/// </para>
/// </summary>
public sealed class YoloPersonDetector : IPersonDetector
{
    private const string ModelUrl =
        "https://huggingface.co/opencv/object_detection_yolox/resolve/main/object_detection_yolox_2022nov.onnx";

    private const int   InputSize   = 640;               // model input H = W = 640
    private const int   NumClasses  = 80;                // COCO
    private const int   PersonClass = 0;                 // COCO "person"
    private const int   Attrs       = 4 + 1 + NumClasses; // cx,cy,w,h,obj + classes = 85
    private const float Pad         = 114f;              // YOLOX letterbox fill value
    private static readonly int[] Strides = [8, 16, 32];

    private readonly string _modelPath;
    private readonly float  _scoreThreshold;
    private readonly float  _nmsThreshold;
    private readonly bool   _useGpu;

    private InferenceSession? _session;
    private string?           _inputName;
    private readonly object   _lock = new();

    /// <inheritdoc />
    public string Name => "YOLOX full-body (COCO person)";

    public YoloPersonDetector(
        string modelsDir,
        float  scoreThreshold = 0.45f,
        float  nmsThreshold   = 0.45f,
        bool   useGpu         = true)
    {
        _modelPath      = Path.Combine(modelsDir, "yolox_2022nov.onnx");
        _scoreThreshold = scoreThreshold;
        _nmsThreshold   = nmsThreshold;
        _useGpu         = useGpu;
        Directory.CreateDirectory(modelsDir);
    }

    // ── Detection ───────────────────────────────────────────────────────────────

    /// <inheritdoc />
    public IReadOnlyList<Detection> Detect(Image<Rgb24> frame)
    {
        EnsureLoaded();

        // Letterbox: scale keeping aspect ratio, paste at the top-left of a
        // 640×640 canvas pre-filled with the padding value (114).
        float r = MathF.Min((float)InputSize / frame.Height, (float)InputSize / frame.Width);
        int newW = Math.Max(1, (int)MathF.Round(frame.Width  * r));
        int newH = Math.Max(1, (int)MathF.Round(frame.Height * r));

        var tensor = new DenseTensor<float>([1, 3, InputSize, InputSize]);
        tensor.Buffer.Span.Fill(Pad);   // letterbox padding (114) over the whole canvas first

        using (Image<Rgb24> resized = frame.Clone(ctx => ctx.Resize(newW, newH)))
        {
            // YOLOX takes BGR pixels in raw 0–255 (no mean/std normalisation); the
            // image is pasted top-left and the rest keeps the padding value.
            resized.ProcessPixelRows(acc =>
            {
                for (int y = 0; y < newH; y++)
                {
                    Span<Rgb24> row = acc.GetRowSpan(y);
                    for (int x = 0; x < newW; x++)
                    {
                        Rgb24 p = row[x];
                        tensor[0, 0, y, x] = p.B;   // channel 0 — B
                        tensor[0, 1, y, x] = p.G;   // channel 1 — G
                        tensor[0, 2, y, x] = p.R;   // channel 2 — R
                    }
                }
            });
        }

        using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs =
            _session!.Run([NamedOnnxValue.CreateFromTensor(_inputName!, tensor)]);

        return Decode(outputs.ToList(), r, frame.Width, frame.Height);
    }

    // ── Output decoding ───────────────────────────────────────────────────────────

    private List<Detection> Decode(
        List<DisposableNamedOnnxValue> outputs, float r, int origW, int origH)
    {
        Tensor<float> outT = outputs[0].AsTensor<float>();
        ReadOnlySpan<int> dims = outT.Dimensions;       // expected [1, 8400, 85]
        float[] data = outT.ToArray();

        // Resolve which axis carries the 85 attributes; support both layouts.
        int n, attrStride;
        if (dims.Length == 3 && dims[2] == Attrs)        // [1, N, 85]
        {
            n = dims[1]; attrStride = 1;
        }
        else if (dims.Length == 3 && dims[1] == Attrs)   // [1, 85, N]
        {
            n = dims[2]; attrStride = dims[2];
        }
        else if (dims.Length == 2 && dims[1] == Attrs)   // [N, 85]
        {
            n = dims[0]; attrStride = 1;
        }
        else
        {
            return [];   // unrecognised output shape
        }

        var candidates = new List<Detection>();
        int anchor = 0;

        // Anchors are concatenated in stride order (8 → 16 → 32), each grid
        // walked row-major (y then x) — matching YOLOX's grid generation.
        foreach (int stride in Strides)
        {
            int g = InputSize / stride;                  // grid side: 80, 40, 20
            for (int gy = 0; gy < g; gy++)
            {
                for (int gx = 0; gx < g; gx++, anchor++)
                {
                    if (anchor >= n) break;

                    float At(int a) => attrStride == 1
                        ? data[anchor * Attrs + a]
                        : data[a * attrStride + anchor];

                    float score = At(4) * At(5 + PersonClass); // objectness × P(person)
                    if (score < _scoreThreshold) continue;

                    // Decode box: grid offset + exp(size), all × stride (padded space).
                    float cx = (At(0) + gx) * stride;
                    float cy = (At(1) + gy) * stride;
                    float w  = MathF.Exp(At(2)) * stride;
                    float h  = MathF.Exp(At(3)) * stride;

                    // Padded space → original image space (top-left letterbox: just / r).
                    float x1 = Math.Clamp((cx - w * 0.5f) / r, 0, origW - 1);
                    float y1 = Math.Clamp((cy - h * 0.5f) / r, 0, origH - 1);
                    float x2 = Math.Clamp((cx + w * 0.5f) / r, 0, origW - 1);
                    float y2 = Math.Clamp((cy + h * 0.5f) / r, 0, origH - 1);

                    candidates.Add(new Detection(x1, y1, x2, y2, score, "person"));
                }
            }
        }

        return Nms.Apply(candidates, _nmsThreshold);
    }

    // ── Model loading ───────────────────────────────────────────────────────────

    private void EnsureLoaded()
    {
        if (_session is not null) return;
        lock (_lock)
        {
            if (_session is not null) return;

            if (!File.Exists(_modelPath) || new FileInfo(_modelPath).Length < 1_000_000)
            {
                Console.WriteLine("  [YOLOX] downloading object_detection_yolox (~36 MB) …");
                Download(ModelUrl, _modelPath);
                Console.WriteLine("  [YOLOX] model ready.");
            }

            _session = CreateSession(_modelPath, _useGpu);
            _inputName = _session.InputMetadata.Keys.First();
            Console.WriteLine($"  [YOLOX] loaded — {_session.OutputMetadata.Count} output(s).");
        }
    }

    // Uses the CUDA execution provider when an NVIDIA GPU + CUDA runtime are
    // actually present, otherwise runs on CPU. The availability probe avoids the
    // noisy native "failed to load onnxruntime_providers_cuda.dll" message on
    // CPU-only machines; any residual failure still falls back to CPU.
    private static InferenceSession CreateSession(string path, bool useGpu)
    {
        if (useGpu && CudaLooksAvailable())
        {
            try
            {
                var gpu = new SessionOptions { LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };
                gpu.AppendExecutionProvider_CUDA(0);
                return new InferenceSession(path, gpu);
            }
            catch
            {
                // fall through to CPU
            }
        }
        var cpu = new SessionOptions { LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };
        return new InferenceSession(path, cpu);
    }

    private static bool CudaLooksAvailable()
    {
        try
        {
            // ORT's CUDA EP needs the CUDA runtime *and* cuDNN; require both so a
            // partial CUDA install doesn't trigger a failed native load.
            GpuStatus s = CudaProvisioner.Detect();
            return s.HasNvidiaGpu && s.CudaRuntimePresent && s.CudnnPresent;
        }
        catch
        {
            return false;
        }
    }

    private static void Download(string url, string dest)
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("PredictiveCoding-Vision/1.0");
        string tmp = dest + ".part";
        using var resp = http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
        resp.EnsureSuccessStatusCode();
        long total = resp.Content.Headers.ContentLength ?? -1;
        long got = 0;
        using (var fs = File.Create(tmp))
        using (var src = resp.Content.ReadAsStreamAsync().GetAwaiter().GetResult())
        {
            byte[] b = new byte[131_072];
            int read;
            while ((read = src.Read(b)) > 0)
            {
                fs.Write(b, 0, read);
                got += read;
                if (total > 0 && got % (1 << 20) < 131_072)
                    Console.Write($"\r  [YOLOX]  {got / 1_048_576:F0} / {total / 1_048_576:F0} MB  ");
            }
        }
        Console.WriteLine();
        File.Move(tmp, dest, overwrite: true);
    }

    public void Dispose() => _session?.Dispose();
}
