using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PredictiveCoding.FaceRecognition;
using PredictiveCoding.FaceRecognition.FaceAlignment;
using PredictiveCoding.Vision.Detectors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using OnnxTensor = PredictiveCoding.Onnx.Tensor;

namespace PredictiveCoding.Onnx.Tests;

/// <summary>
/// The in-house runtime wired into the real detectors. Raw-output parity for the two
/// models not covered by <see cref="OnnxYoloxParityTests"/> (RetinaFace det_10g,
/// ArcFace w600k_r50) plus wiring smoke tests for the useInHouseRuntime flag.
/// Each test skips silently (returns) when its model is not on disk — the repo
/// downloads models on first use, so CI machines may not have them.
/// </summary>
public class OnnxDetectorWiringTests
{
    private static readonly string ModelsDir = Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".predictivecoding", "vision-models");

    // ── Raw parity: in-house executor vs ONNX Runtime ────────────────────────

    [Fact]
    [Trait("Category", "Integration")]
    public void RetinaFace_InHouse_MatchesOnnxRuntime()
    {
        string path = Path.Combine(ModelsDir, "det_10g.onnx");
        if (!File.Exists(path)) return;   // model not downloaded on this machine

        // det_10g has dynamic H/W — run at the canonical 640×640, in-distribution
        // input ((pixel−127.5)/128 ∈ [−1,1]).
        AssertParity(path, [1, 3, 640, 640], maxAbs: 1e-3, meanAbs: 1e-4);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ArcFace_InHouse_MatchesOnnxRuntime()
    {
        string path = Path.Combine(ModelsDir, "arcface_w600k_r50.onnx");
        if (!File.Exists(path)) return;

        AssertParity(path, [1, 3, 112, 112], maxAbs: 1e-3, meanAbs: 1e-4, minCosine: 0.99999);
    }

    private static void AssertParity(string path, int[] shape, double maxAbs, double meanAbs,
                                     double minCosine = 0.999)
    {
        var rng = new Random(42);
        var data = new float[OnnxTensor.CheckedProduct(shape)];
        for (int i = 0; i < data.Length; i++) data[i] = (float)(rng.NextDouble() * 2.0 - 1.0);

        var model = OnnxModel.Load(path);
        var mine = model.Run(new() { [model.InputNames[0]] = new OnnxTensor(shape, data) });

        using var session = new InferenceSession(path);
        string inputName = session.InputMetadata.Keys.First();
        using var ortOut = session.Run([NamedOnnxValue.CreateFromTensor(inputName, new DenseTensor<float>(data, shape))]);

        foreach (var o in ortOut)
        {
            Assert.True(mine.ContainsKey(o.Name), $"in-house run missing output '{o.Name}'");
            float[] theirs = o.AsEnumerable<float>().ToArray();
            float[] ours = mine[o.Name].Data;
            Assert.Equal(theirs.Length, ours.Length);

            double max = 0, sum = 0, dot = 0, na = 0, nb = 0;
            for (int i = 0; i < theirs.Length; i++)
            {
                double d = Math.Abs(theirs[i] - ours[i]);
                if (d > max) max = d;
                sum += d;
                dot += (double)theirs[i] * ours[i];
                na += (double)theirs[i] * theirs[i];
                nb += (double)ours[i] * ours[i];
            }
            double mean = sum / theirs.Length;
            double cos = dot / Math.Max(Math.Sqrt(na) * Math.Sqrt(nb), 1e-12);
            Assert.True(max < maxAbs && mean < meanAbs && cos > minCosine,
                $"output '{o.Name}': maxAbsDiff={max:e3} (limit {maxAbs:e1}), meanAbsDiff={mean:e3} (limit {meanAbs:e1}), cosine={cos:F7} (min {minCosine})");
        }
    }

    // ── Wiring smoke tests: the flag actually drives the in-house path ───────

    [Fact]
    [Trait("Category", "Integration")]
    public void YoloPersonDetector_InHouseRuntime_Runs()
    {
        if (!File.Exists(Path.Combine(ModelsDir, "yolox_2022nov.onnx"))) return;

        using var detector = new YoloPersonDetector(ModelsDir, useGpu: false, useInHouseRuntime: true);
        Assert.Contains("in-house", detector.Name);

        using var frame = Gradient(320, 240);
        var detections = detector.Detect(frame);

        Assert.NotNull(detections);   // synthetic gradient: no people expected, but the pipeline must run
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void RetinaFaceDetector_InHouseRuntime_Runs()
    {
        if (!File.Exists(Path.Combine(ModelsDir, "det_10g.onnx"))) return;

        using var detector = new RetinaFaceDetector(ModelsDir, useInHouseRuntime: true);
        using var frame = Gradient(160, 160);
        var faces = detector.DetectAll(frame);

        Assert.NotNull(faces);
    }

    [Fact]
    [Trait("Category", "Integration")]
    public void ArcFaceEmbedder_InHouseRuntime_ProducesUnitEmbedding()
    {
        if (!File.Exists(Path.Combine(ModelsDir, "arcface_w600k_r50.onnx"))) return;

        using var embedder = new ArcFaceOnnxEmbedder(ModelsDir, useAlignment: false, useInHouseRuntime: true);
        string tmp = Path.Combine(Path.GetTempPath(), $"pc-onnx-face-{Guid.NewGuid():N}.png");
        try
        {
            using (var img = Gradient(64, 64)) img.SaveAsPng(tmp);
            float[] v = embedder.Embed(tmp);

            Assert.Equal(512, v.Length);
            double norm = 0;
            foreach (float x in v) norm += (double)x * x;
            Assert.Equal(1.0, Math.Sqrt(norm), 3);   // embedder L2-normalises
        }
        finally
        {
            File.Delete(tmp);
        }
    }

    private static Image<Rgb24> Gradient(int w, int h)
    {
        var img = new Image<Rgb24>(w, h);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                img[x, y] = new Rgb24((byte)(x * 255 / w), (byte)(y * 255 / h), 128);
        return img;
    }
}
