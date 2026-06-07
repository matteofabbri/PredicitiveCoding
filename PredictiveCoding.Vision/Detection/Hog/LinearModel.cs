using System.Globalization;

namespace PredictiveCoding.Vision.Detectors;

/// <summary>
/// A linear classifier (weight vector + bias) over feature vectors — the "SVM" head
/// of the in-house HOG detector. Trained in-house by sub-gradient descent on the
/// hinge loss (Pegasos-style), so the whole detector is ours end to end; the score is
/// just w·x + b. Weights persist to / load from a small text file.
/// </summary>
public sealed class LinearModel
{
    /// <summary>Weight vector (length = feature dimension).</summary>
    public float[] Weights { get; }

    /// <summary>Bias term.</summary>
    public float Bias { get; set; }

    /// <summary>Class label the positive side represents (e.g. "person").</summary>
    public string Label { get; set; }

    public int Dimension => Weights.Length;

    public LinearModel(int dimension, string label = "object")
    {
        if (dimension < 1) throw new ArgumentOutOfRangeException(nameof(dimension));
        Weights = new float[dimension];
        Label = label;
    }

    private LinearModel(float[] weights, float bias, string label)
    {
        Weights = weights;
        Bias = bias;
        Label = label;
    }

    /// <summary>Raw decision score w·x + b.</summary>
    public float Score(ReadOnlySpan<float> features)
    {
        if (features.Length != Weights.Length)
            throw new ArgumentException($"Expected {Weights.Length} features, got {features.Length}.");
        float s = Bias;
        for (int i = 0; i < Weights.Length; i++) s += Weights[i] * features[i];
        return s;
    }

    /// <summary>True when the score is at or above <paramref name="threshold"/>.</summary>
    public bool Predict(ReadOnlySpan<float> features, float threshold = 0f) => Score(features) >= threshold;

    /// <summary>One Pegasos sub-gradient step for a single sample.
    /// <paramref name="label"/> must be +1 (positive) or −1 (negative).</summary>
    public void TrainStep(ReadOnlySpan<float> features, int label, float learningRate, float l2 = 1e-4f)
    {
        float margin = label * Score(features);
        // L2 shrinkage applies every step.
        for (int i = 0; i < Weights.Length; i++) Weights[i] *= 1f - learningRate * l2;
        if (margin < 1f)
        {
            for (int i = 0; i < Weights.Length; i++) Weights[i] += learningRate * label * features[i];
            Bias += learningRate * label;
        }
    }

    /// <summary>Trains on labelled samples for several epochs (shuffled each epoch).</summary>
    public void Fit(IReadOnlyList<(float[] Features, int Label)> samples,
        int epochs = 10, float learningRate = 0.01f, float l2 = 1e-4f, int seed = 1)
    {
        ArgumentNullException.ThrowIfNull(samples);
        var order = Enumerable.Range(0, samples.Count).ToArray();
        var rng = new Random(seed);

        for (int e = 0; e < epochs; e++)
        {
            for (int i = order.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }
            foreach (int k in order)
            {
                var (f, y) = samples[k];
                TrainStep(f, y >= 0 ? 1 : -1, learningRate, l2);
            }
        }
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    /// <summary>Saves the model as text: header line then the weights.</summary>
    public void Save(TextWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);
        writer.WriteLine($"hoglinear v1 {Label} {Dimension.ToString(CultureInfo.InvariantCulture)} {Bias.ToString(CultureInfo.InvariantCulture)}");
        foreach (float w in Weights) writer.WriteLine(w.ToString(CultureInfo.InvariantCulture));
    }

    /// <summary>Saves the model to a file.</summary>
    public void SaveToFile(string path)
    {
        using var w = new StreamWriter(path);
        Save(w);
    }

    /// <summary>Reads a model written by <see cref="Save"/>.</summary>
    public static LinearModel Load(TextReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        string header = reader.ReadLine() ?? throw new FormatException("Empty model.");
        string[] h = header.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (h.Length != 5 || h[0] != "hoglinear")
            throw new FormatException($"Bad model header: '{header}'.");

        string label = h[2];
        int dim = int.Parse(h[3], CultureInfo.InvariantCulture);
        float bias = float.Parse(h[4], CultureInfo.InvariantCulture);

        var weights = new float[dim];
        for (int i = 0; i < dim; i++)
            weights[i] = float.Parse(
                reader.ReadLine() ?? throw new FormatException("Truncated model weights."),
                CultureInfo.InvariantCulture);

        return new LinearModel(weights, bias, label);
    }

    /// <summary>Loads a model from a file.</summary>
    public static LinearModel LoadFromFile(string path)
    {
        using var r = new StreamReader(path);
        return Load(r);
    }
}
