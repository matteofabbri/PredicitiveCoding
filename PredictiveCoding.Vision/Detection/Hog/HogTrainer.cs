using System.Drawing;
using PredictiveCoding.Imaging;

namespace PredictiveCoding.Vision.Detectors;

/// <summary>
/// Builds a HOG training set and fits the in-house <see cref="LinearModel"/>. Positives
/// are crops the teacher (e.g. YOLOX / RetinaFace) flagged as the object; negatives are
/// background windows. This is the in-house, end-to-end training behind the detector —
/// the ONNX models only supply the labels (knowledge distillation), and once trained the
/// detector runs without them.
/// </summary>
public sealed class HogTrainer
{
    private readonly HogDescriptor _hog;
    private readonly List<(float[] Features, int Label)> _samples = [];

    public int PositiveCount { get; private set; }
    public int NegativeCount { get; private set; }
    public int SampleCount => _samples.Count;

    public HogTrainer(HogDescriptor hog)
    {
        ArgumentNullException.ThrowIfNull(hog);
        _hog = hog;
    }

    /// <summary>Adds a positive sample from a teacher detection box.</summary>
    public void AddPositive(RgbImage image, Rectangle box)
    {
        _samples.Add((_hog.FromCrop(image, box), +1));
        PositiveCount++;
    }

    /// <summary>Adds a negative sample from a background box.</summary>
    public void AddNegative(RgbImage image, Rectangle box)
    {
        _samples.Add((_hog.FromCrop(image, box), -1));
        NegativeCount++;
    }

    /// <summary>Adds a pre-computed feature/label sample directly.</summary>
    public void AddSample(float[] features, int label)
    {
        ArgumentNullException.ThrowIfNull(features);
        if (features.Length != _hog.Dimension)
            throw new ArgumentException("Feature length mismatch.", nameof(features));
        _samples.Add((features, label >= 0 ? 1 : -1));
        if (label >= 0) PositiveCount++; else NegativeCount++;
    }

    /// <summary>Samples <paramref name="count"/> random background windows from
    /// <paramref name="image"/> that do not overlap any box in <paramref name="avoid"/>
    /// (IoU ≤ <paramref name="maxIou"/>), adding them as negatives. Returns the number added.</summary>
    public int AddRandomNegatives(RgbImage image, IReadOnlyList<Rectangle> avoid,
        int count, Random rng, float maxIou = 0.1f, int maxAttempts = 0)
    {
        ArgumentNullException.ThrowIfNull(image);
        ArgumentNullException.ThrowIfNull(avoid);
        ArgumentNullException.ThrowIfNull(rng);
        if (image.Width < _hog.WindowWidth || image.Height < _hog.WindowHeight) return 0;

        float aspect = (float)_hog.WindowWidth / _hog.WindowHeight;
        int added = 0, attempts = 0, limit = maxAttempts > 0 ? maxAttempts : count * 20;

        while (added < count && attempts++ < limit)
        {
            int h = rng.Next(_hog.WindowHeight, image.Height + 1);
            int w = Math.Clamp((int)(h * aspect), _hog.WindowWidth, image.Width);
            int x = rng.Next(0, image.Width - w + 1);
            int y = rng.Next(0, image.Height - h + 1);
            var box = new Rectangle(x, y, w, h);

            bool clashes = false;
            foreach (Rectangle a in avoid)
                if (RectIou(box, a) > maxIou) { clashes = true; break; }
            if (clashes) continue;

            AddNegative(image, box);
            added++;
        }
        return added;
    }

    /// <summary>Fits a linear model on all accumulated samples.</summary>
    public LinearModel Train(string label, int epochs = 15, float learningRate = 0.01f, float l2 = 1e-4f, int seed = 1)
    {
        if (_samples.Count == 0)
            throw new InvalidOperationException("No samples to train on.");
        var model = new LinearModel(_hog.Dimension, label);
        model.Fit(_samples, epochs, learningRate, l2, seed);
        return model;
    }

    private static float RectIou(Rectangle a, Rectangle b)
    {
        int ix1 = Math.Max(a.Left, b.Left), iy1 = Math.Max(a.Top, b.Top);
        int ix2 = Math.Min(a.Right, b.Right), iy2 = Math.Min(a.Bottom, b.Bottom);
        int inter = Math.Max(0, ix2 - ix1) * Math.Max(0, iy2 - iy1);
        int union = a.Width * a.Height + b.Width * b.Height - inter;
        return union > 0 ? (float)inter / union : 0f;
    }
}
