using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PredictiveCoding.FaceRecognition.FaceAlignment;

/// <summary>
/// ONNX-based face detector using the InsightFace buffalo_l <c>det_10g.onnx</c> model
/// (RetinaFace / SCRFD architecture, 10 G MACs).
/// <para>
/// Detects the most-prominent face in an image and returns its bounding box plus
/// 5 facial landmarks (left eye, right eye, nose tip, left mouth, right mouth)
/// in <b>original image</b> coordinates.
/// </para>
/// Model is downloaded automatically (~16 MB) from HuggingFace on first use.
/// </summary>
public sealed class RetinaFaceDetector : IDisposable
{
    private const string ModelUrl =
        "https://huggingface.co/immich-app/buffalo_l/resolve/main/det_10g.onnx";

    private const int InputSize      = 640;   // model input (H = W = 640)
    private const float ScoreThresh  = 0.5f;
    private const float NmsThresh    = 0.4f;
    private const int NumAnchors     = 2;     // anchor count per spatial location
    private static readonly int[] Strides = [8, 16, 32];

    // Expected N (number of anchors) at each stride for InputSize × InputSize
    private static readonly HashSet<int> KnownN =
    [
        InputSize / 8  * InputSize / 8  * NumAnchors,   // 12 800 (stride 8)
        InputSize / 16 * InputSize / 16 * NumAnchors,   // 3 200  (stride 16)
        InputSize / 32 * InputSize / 32 * NumAnchors,   // 800    (stride 32)
    ];

    private readonly string  _modelPath;
    private InferenceSession? _session;
    private string?           _inputName;
    private readonly object   _lock = new();

    public RetinaFaceDetector(string modelsDir)
    {
        _modelPath = Path.Combine(modelsDir, "det_10g.onnx");
        Directory.CreateDirectory(modelsDir);
    }

    // ── Detection ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Returns the highest-confidence face in the image, or <see langword="null"/> if none found.
    /// All coordinates in the result are in original image space.
    /// </summary>
    public FaceDetection? Detect(string imagePath)
    {
        using Image<Rgb24> img = Image.Load<Rgb24>(imagePath);
        return Detect(img);
    }

    /// <summary>
    /// Returns the highest-confidence face in <paramref name="image"/>, or <see langword="null"/> if none found.
    /// The method does NOT dispose <paramref name="image"/>.
    /// </summary>
    public FaceDetection? Detect(Image<Rgb24> image)
    {
        EnsureLoaded();

        // Scale factors: convert 640×640 detection coords → original image coords
        float scaleX = (float)image.Width  / InputSize;
        float scaleY = (float)image.Height / InputSize;

        // Build NCHW float32 tensor: (R−127.5)/128, (G−127.5)/128, (B−127.5)/128
        var tensor = new DenseTensor<float>([1, 3, InputSize, InputSize]);
        using (var resized = image.Clone(ctx => ctx.Resize(InputSize, InputSize)))
        {
            resized.ProcessPixelRows(acc =>
            {
                for (int y = 0; y < InputSize; y++)
                {
                    Span<Rgb24> row = acc.GetRowSpan(y);
                    for (int x = 0; x < InputSize; x++)
                    {
                        Rgb24 p = row[x];
                        tensor[0, 0, y, x] = (p.R - 127.5f) / 128f;
                        tensor[0, 1, y, x] = (p.G - 127.5f) / 128f;
                        tensor[0, 2, y, x] = (p.B - 127.5f) / 128f;
                    }
                }
            });
        }

        using var outputs = _session!.Run(
            [NamedOnnxValue.CreateFromTensor(_inputName!, tensor)]);

        return ParseAndDecode(outputs.ToList(), scaleX, scaleY);
    }

    // ── Output parsing ────────────────────────────────────────────────────────────

    private static FaceDetection? ParseAndDecode(
        IReadOnlyList<NamedOnnxValue> outputs,
        float scaleX, float scaleY)
    {
        // Classify the 9 output tensors by shape:
        // lastDim == 1 → classification scores
        // lastDim == 4 → bounding-box distances
        // lastDim == 10 → keypoint offsets (5 × 2)
        // Within each group sort by N descending → stride 8 (N=12800), 16, 32.

        var scoreOuts = new List<(int N, float[] Data)>();
        var bboxOuts  = new List<(int N, float[] Data)>();
        var kpsOuts   = new List<(int N, float[] Data)>();

        foreach (NamedOnnxValue ov in outputs)
        {
            try
            {
                Tensor<float> t = ov.AsTensor<float>();
                ReadOnlySpan<int> dims = t.Dimensions;
                float[] data = t.ToArray();

                // Determine N and lastDim from tensor shape
                int n, lastDim;
                if (dims.Length == 3)
                {
                    // [batch=1, N, lastDim]
                    n = dims[1]; lastDim = dims[2];
                }
                else if (dims.Length == 2)
                {
                    // [1, N] → scores,  or  [N, lastDim]
                    if (dims[0] == 1 && KnownN.Contains(dims[1]))
                        { n = dims[1]; lastDim = 1; }
                    else
                        { n = dims[0]; lastDim = dims[1]; }
                }
                else if (dims.Length == 1)
                {
                    n = dims[0]; lastDim = 1;
                }
                else continue;

                if (!KnownN.Contains(n)) continue;

                switch (lastDim)
                {
                    case  1: scoreOuts.Add((n, data)); break;
                    case  4: bboxOuts .Add((n, data)); break;
                    case 10: kpsOuts  .Add((n, data)); break;
                }
            }
            catch { /* skip unrecognised outputs */ }
        }

        // Sort by N descending (stride 8 first → N=12800, 16 → 3200, 32 → 800)
        scoreOuts.Sort((a, b) => b.N.CompareTo(a.N));
        bboxOuts .Sort((a, b) => b.N.CompareTo(a.N));
        kpsOuts  .Sort((a, b) => b.N.CompareTo(a.N));

        if (scoreOuts.Count < 3 || bboxOuts.Count < 3 || kpsOuts.Count < 3)
            return null; // model format not recognised

        // Collect candidates
        var candidates = new List<(float Score, float X1, float Y1, float X2, float Y2, float[] Kps)>();

        for (int si = 0; si < Strides.Length; si++)
        {
            int stride = Strides[si];
            int w = InputSize / stride;           // feature map width
            int n = scoreOuts[si].N;              // 12800 / 3200 / 800

            float[] scores = scoreOuts[si].Data;
            float[] bbox   = bboxOuts [si].Data;
            float[] kps    = kpsOuts  [si].Data;

            for (int a = 0; a < n; a++)
            {
                float score = Sigmoid(scores[a]);
                if (score < ScoreThresh) continue;

                // Anchor centre (in InputSize×InputSize coordinates)
                int  spatialIdx = a / NumAnchors;
                int  gridI      = spatialIdx / w;       // row
                int  gridJ      = spatialIdx % w;       // column
                float cx = gridJ * stride;
                float cy = gridI * stride;

                // Decode bounding box (FCOS-style distance from centre, pixel units)
                float x1 = (cx - bbox[a * 4 + 0]) * scaleX;
                float y1 = (cy - bbox[a * 4 + 1]) * scaleY;
                float x2 = (cx + bbox[a * 4 + 2]) * scaleX;
                float y2 = (cy + bbox[a * 4 + 3]) * scaleY;

                // Decode 5 keypoints (offsets from anchor centre)
                var lmks = new float[10];
                for (int k = 0; k < 5; k++)
                {
                    lmks[k * 2]     = (cx + kps[a * 10 + k * 2])     * scaleX;
                    lmks[k * 2 + 1] = (cy + kps[a * 10 + k * 2 + 1]) * scaleY;
                }

                candidates.Add((score, x1, y1, x2, y2, lmks));
            }
        }

        if (candidates.Count == 0) return null;

        // Greedy NMS — keep highest-scoring non-overlapping detections
        candidates.Sort((a, b) => b.Score.CompareTo(a.Score));
        var kept = new List<int> { 0 };
        for (int i = 1; i < candidates.Count; i++)
        {
            bool suppressed = false;
            foreach (int j in kept)
            {
                if (Iou(candidates[i], candidates[j]) > NmsThresh)
                { suppressed = true; break; }
            }
            if (!suppressed) kept.Add(i);
        }

        var best = candidates[kept[0]];
        return new FaceDetection(best.X1, best.Y1, best.X2, best.Y2, best.Score, best.Kps);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────────

    private static float Sigmoid(float x) => 1f / (1f + MathF.Exp(-x));

    private static float Iou(
        (float Score, float X1, float Y1, float X2, float Y2, float[] Kps) a,
        (float Score, float X1, float Y1, float X2, float Y2, float[] Kps) b)
    {
        float ix1 = MathF.Max(a.X1, b.X1); float iy1 = MathF.Max(a.Y1, b.Y1);
        float ix2 = MathF.Min(a.X2, b.X2); float iy2 = MathF.Min(a.Y2, b.Y2);
        float inter = MathF.Max(0f, ix2 - ix1) * MathF.Max(0f, iy2 - iy1);
        float areaA = (a.X2 - a.X1) * (a.Y2 - a.Y1);
        float areaB = (b.X2 - b.X1) * (b.Y2 - b.Y1);
        float union = areaA + areaB - inter;
        return union > 0 ? inter / union : 0f;
    }

    // ── Model loading ─────────────────────────────────────────────────────────────

    private void EnsureLoaded()
    {
        if (_session is not null) return;
        lock (_lock)
        {
            if (_session is not null) return;

            if (!File.Exists(_modelPath) || new FileInfo(_modelPath).Length < 1_000_000)
            {
                Console.WriteLine("  [RetinaFace] downloading det_10g.onnx (~16 MB) …");
                Download(ModelUrl, _modelPath);
                Console.WriteLine("  [RetinaFace] model ready.");
            }

            var options = new SessionOptions { LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR };
            _session   = new InferenceSession(_modelPath, options);
            _inputName = _session.InputMetadata.Keys.First();

            Console.WriteLine($"  [RetinaFace] loaded — {_session.OutputMetadata.Count} outputs");
        }
    }

    private static void Download(string url, string dest)
    {
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        http.DefaultRequestHeaders.UserAgent.ParseAdd("PredictiveCoding-Face/1.0");
        string tmp = dest + ".part";
        using var resp = http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead).GetAwaiter().GetResult();
        resp.EnsureSuccessStatusCode();
        long   total = resp.Content.Headers.ContentLength ?? -1;
        long   got   = 0;
        using (var fs  = File.Create(tmp))
        using (var src = resp.Content.ReadAsStreamAsync().GetAwaiter().GetResult())
        {
            byte[] buf = new byte[131_072];
            int read;
            while ((read = src.Read(buf)) > 0)
            {
                fs.Write(buf, 0, read);
                got += read;
                if (total > 0 && got % (1 << 20) < 131_072)
                    Console.Write($"\r  [RetinaFace]  {got / 1_048_576:F0} / {total / 1_048_576:F0} MB  ");
            }
        }
        Console.WriteLine();
        File.Move(tmp, dest, overwrite: true);
    }

    public void Dispose() => _session?.Dispose();
}
