using PredictiveCoding.Vision.Detectors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using Xunit;

namespace PredictiveCoding.Vision.Tests;

/// <summary>Tests for the in-house HOG + linear-model detector and its trainer
/// (no external model or ONNX runtime). Synthetic images keep everything deterministic.</summary>
public class HogDetectorTests
{
    private static readonly Rgb24 Black = new(0, 0, 0);
    private static readonly Rgb24 White = new(255, 255, 255);

    private static Image<Rgb24> Fill(int w, int h, Func<int, int, Rgb24> f)
    {
        var img = new Image<Rgb24>(w, h);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                img[x, y] = f(x, y);
        return img;
    }

    private static double Dot(float[] a, float[] b)
    {
        double s = 0;
        for (int i = 0; i < a.Length; i++) s += (double)a[i] * b[i];
        return s;
    }

    // ── HogDescriptor ─────────────────────────────────────────────────────────

    [Fact]
    public void Hog_Dimension_MatchesDalalTriggsLayout()
        => Assert.Equal(7 * 15 * 36, new HogDescriptor(64, 128, 8, 9).Dimension);

    [Fact]
    public void Hog_FlatImage_IsAllZero_EdgedImage_IsNot()
    {
        var hog = new HogDescriptor(64, 128);

        using var flat = Fill(64, 128, (_, _) => new Rgb24(128, 128, 128));
        float[] f0 = hog.FromWindow(flat, 0, 0);
        Assert.Equal(hog.Dimension, f0.Length);
        Assert.All(f0, v => Assert.Equal(0f, v));               // no gradients → zero HOG

        using var edged = Fill(64, 128, (x, _) => x < 32 ? Black : White);
        float[] f1 = hog.FromWindow(edged, 0, 0);
        Assert.All(f1, v => Assert.True(float.IsFinite(v)));
        Assert.True(Dot(f1, f1) > 0, "an edge must produce non-zero HOG energy");
    }

    // ── LinearModel ───────────────────────────────────────────────────────────

    [Fact]
    public void LinearModel_LearnsSeparableClasses_AndRoundTrips()
    {
        var rng = new Random(0);
        var samples = new List<(float[], int)>();
        for (int i = 0; i < 40; i++)
        {
            float n() => (float)(rng.NextDouble() * 0.2 - 0.1);
            samples.Add(([1 + n(), 1 + n(), n(), n()], +1));
            samples.Add(([n(), n(), 1 + n(), 1 + n()], -1));
        }

        var model = new LinearModel(4, "demo");
        model.Fit(samples, epochs: 60, learningRate: 0.05f);

        Assert.True(model.Score([1, 1, 0, 0]) > 0);
        Assert.True(model.Score([0, 0, 1, 1]) < 0);

        var sw = new StringWriter();
        model.Save(sw);
        var reloaded = LinearModel.Load(new StringReader(sw.ToString()));
        Assert.Equal("demo", reloaded.Label);
        Assert.Equal(model.Bias, reloaded.Bias, 5);
        Assert.Equal(model.Weights, reloaded.Weights);
    }

    // ── HogPersonDetector ─────────────────────────────────────────────────────

    [Fact]
    public void Detector_FindsTheTemplateWindow()
    {
        var hog = new HogDescriptor(64, 128);
        using var img = Fill(64, 128, (x, y) => new Rgb24((byte)((x * 7 + y * 13) % 256), (byte)((y * 5) % 256), (byte)((x * 3) % 256)));

        // A template-matching model: weights = the window's own HOG.
        float[] template = hog.FromWindow(img, 0, 0);
        var model = new LinearModel(hog.Dimension, "thing");
        Array.Copy(template, model.Weights, template.Length);
        model.Bias = (float)(-0.5 * Dot(template, template));   // score at the template = ½‖t‖² > 0

        var detector = new HogPersonDetector(hog, model, scoreThreshold: 0f, pyramidScale: 1.5f);
        var dets = detector.Detect(img);

        Assert.Single(dets);
        Assert.Equal(0f, dets[0].X1, 1f);
        Assert.Equal(0f, dets[0].Y1, 1f);
        Assert.Equal(64f, dets[0].X2, 1f);
        Assert.Equal(128f, dets[0].Y2, 1f);
        Assert.Equal("thing", dets[0].Label);
    }

    // ── HogTrainer ────────────────────────────────────────────────────────────

    [Fact]
    public void Trainer_SeparatesPositiveFromNegativeAppearance()
    {
        var hog = new HogDescriptor(32, 32);
        using var vertical = Fill(64, 64, (x, _) => (x / 4) % 2 == 0 ? Black : White);   // vertical stripes
        using var horizontal = Fill(64, 64, (_, y) => (y / 4) % 2 == 0 ? Black : White); // horizontal stripes
        var whole = new Rectangle(0, 0, 64, 64);

        var trainer = new HogTrainer(hog);
        for (int i = 0; i < 12; i++) trainer.AddPositive(vertical, whole);
        for (int i = 0; i < 12; i++) trainer.AddNegative(horizontal, whole);
        Assert.Equal(24, trainer.SampleCount);

        var model = trainer.Train("vstripes", epochs: 40, learningRate: 0.05f);
        float sPos = model.Score(hog.FromCrop(vertical, whole));
        float sNeg = model.Score(hog.FromCrop(horizontal, whole));

        Assert.True(sPos > 0, $"positive score {sPos}");
        Assert.True(sNeg < 0, $"negative score {sNeg}");
    }

    [Fact]
    public void Trainer_AddRandomNegatives_AvoidsGivenBoxes()
    {
        var hog = new HogDescriptor(32, 32);
        using var img = Fill(200, 200, (x, y) => new Rgb24((byte)x, (byte)y, 0));
        var trainer = new HogTrainer(hog);

        int added = trainer.AddRandomNegatives(img, [new Rectangle(0, 0, 60, 120)], 10, new Random(1));

        Assert.True(added >= 1);
        Assert.Equal(added, trainer.NegativeCount);
        Assert.Equal(added, trainer.SampleCount);
    }

    // ── LinearRegressor (teacher distillation) ────────────────────────────────

    [Fact]
    public void LinearRegressor_LearnsLinearMap_AndRoundTrips()
    {
        var rng = new Random(2);
        var samples = new List<(float[], float[])>();
        for (int i = 0; i < 200; i++)
        {
            float a = (float)rng.NextDouble(), b = (float)rng.NextDouble(), c = (float)rng.NextDouble();
            samples.Add(([a, b, c], [a + 2 * b, -c]));     // the target map to recover
        }

        var reg = new LinearRegressor(3, 2);
        reg.Fit(samples, epochs: 200, learningRate: 0.05f);

        float[] y = reg.Predict([0.5f, 0.5f, 0.5f]);
        Assert.Equal(1.5f, y[0], 0.15f);
        Assert.Equal(-0.5f, y[1], 0.15f);

        var sw = new StringWriter();
        reg.Save(sw);
        var reloaded = LinearRegressor.Load(new StringReader(sw.ToString()));
        Assert.Equal(reg.Predict([0.5f, 0.5f, 0.5f]), reloaded.Predict([0.5f, 0.5f, 0.5f]));
    }
}
