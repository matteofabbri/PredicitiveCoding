using System.Globalization;

namespace PredictiveCoding.Vision.Detectors;

/// <summary>
/// A multi-output linear regressor (W·x + b) trained in-house by SGD on the mean-
/// squared error. Used to <b>distil</b> a teacher embedder (e.g. ArcFace) into an
/// in-house one: regress a face's HOG features onto the teacher's embedding, and at
/// inference produce a comparable vector without the ONNX model. Persists to text.
/// </summary>
public sealed class LinearRegressor
{
    private readonly float[][] _w;   // [outputDim][inputDim]
    private readonly float[] _b;     // [outputDim]

    public int InputDim { get; }
    public int OutputDim { get; }

    public LinearRegressor(int inputDim, int outputDim)
    {
        if (inputDim < 1) throw new ArgumentOutOfRangeException(nameof(inputDim));
        if (outputDim < 1) throw new ArgumentOutOfRangeException(nameof(outputDim));
        InputDim = inputDim;
        OutputDim = outputDim;
        _w = new float[outputDim][];
        for (int o = 0; o < outputDim; o++) _w[o] = new float[inputDim];
        _b = new float[outputDim];
    }

    private LinearRegressor(float[][] w, float[] b)
    {
        _w = w;
        _b = b;
        OutputDim = w.Length;
        InputDim = w.Length > 0 ? w[0].Length : 0;
    }

    /// <summary>Predicts the output vector for <paramref name="x"/>.</summary>
    public float[] Predict(ReadOnlySpan<float> x)
    {
        if (x.Length != InputDim)
            throw new ArgumentException($"Expected {InputDim} inputs, got {x.Length}.");
        var y = new float[OutputDim];
        for (int o = 0; o < OutputDim; o++)
        {
            float s = _b[o];
            float[] wo = _w[o];
            for (int i = 0; i < InputDim; i++) s += wo[i] * x[i];
            y[o] = s;
        }
        return y;
    }

    /// <summary>Predicts and L2-normalises the output (for cosine comparison, like ArcFace).</summary>
    public float[] PredictNormalized(ReadOnlySpan<float> x)
    {
        var y = Predict(x);
        double norm = 0;
        foreach (float v in y) norm += (double)v * v;
        norm = Math.Sqrt(norm);
        if (norm > 1e-8)
            for (int i = 0; i < y.Length; i++) y[i] = (float)(y[i] / norm);
        return y;
    }

    /// <summary>One SGD step minimising ‖W·x + b − target‖².</summary>
    public void TrainStep(ReadOnlySpan<float> x, ReadOnlySpan<float> target, float learningRate, float l2 = 1e-5f)
    {
        if (x.Length != InputDim) throw new ArgumentException("Input length mismatch.", nameof(x));
        if (target.Length != OutputDim) throw new ArgumentException("Target length mismatch.", nameof(target));

        var pred = Predict(x);
        for (int o = 0; o < OutputDim; o++)
        {
            float err = pred[o] - target[o];
            float[] wo = _w[o];
            for (int i = 0; i < InputDim; i++)
                wo[i] -= learningRate * (err * x[i] + l2 * wo[i]);
            _b[o] -= learningRate * err;
        }
    }

    /// <summary>Trains on (input, target) pairs for several shuffled epochs.</summary>
    public void Fit(IReadOnlyList<(float[] Input, float[] Target)> samples,
        int epochs = 20, float learningRate = 0.001f, float l2 = 1e-5f, int seed = 1)
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
                TrainStep(samples[k].Input, samples[k].Target, learningRate, l2);
        }
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    public void Save(TextWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);
        writer.WriteLine($"linreg v1 {InputDim.ToString(CultureInfo.InvariantCulture)} {OutputDim.ToString(CultureInfo.InvariantCulture)}");
        for (int o = 0; o < OutputDim; o++)
        {
            writer.Write(_b[o].ToString(CultureInfo.InvariantCulture));
            foreach (float w in _w[o]) { writer.Write(' '); writer.Write(w.ToString(CultureInfo.InvariantCulture)); }
            writer.WriteLine();
        }
    }

    public void SaveToFile(string path)
    {
        using var w = new StreamWriter(path);
        Save(w);
    }

    public static LinearRegressor Load(TextReader reader)
    {
        ArgumentNullException.ThrowIfNull(reader);
        string[] h = (reader.ReadLine() ?? throw new FormatException("Empty regressor."))
            .Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (h.Length != 4 || h[0] != "linreg") throw new FormatException("Bad regressor header.");
        int inDim = int.Parse(h[2], CultureInfo.InvariantCulture);
        int outDim = int.Parse(h[3], CultureInfo.InvariantCulture);

        var w = new float[outDim][];
        var b = new float[outDim];
        for (int o = 0; o < outDim; o++)
        {
            string[] row = (reader.ReadLine() ?? throw new FormatException("Truncated regressor."))
                .Split(' ', StringSplitOptions.RemoveEmptyEntries);
            b[o] = float.Parse(row[0], CultureInfo.InvariantCulture);
            w[o] = new float[inDim];
            for (int i = 0; i < inDim; i++) w[o][i] = float.Parse(row[i + 1], CultureInfo.InvariantCulture);
        }
        return new LinearRegressor(w, b);
    }
}
