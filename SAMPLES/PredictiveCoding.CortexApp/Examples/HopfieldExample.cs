using PredictiveCoding.Core;
using PredictiveCoding.Expanded.Hopfield;
using PredictiveCoding.Networks;

namespace PredictiveCoding.CortexApp.Examples;

// Trains a Hopfield-augmented generative network on a few 12×12 prototype shapes,
// then corrupts one and completes it from the partial/noisy cue.
public static class HopfieldExample
{
    public const int Size = 12;          // 12×12 patterns
    private const int Dim  = Size * Size;

    public sealed record Result(
        DenseTensor<float> Original,
        DenseTensor<float> Corrupted,
        DenseTensor<float> Recalled,
        float              Similarity);

    public static Result Run(int epochs, int steps)
    {
        var rng = new Random(7);
        var cfg = new HopfieldNetworkConfig
        {
            NetworkConfig       = new NetworkConfig { LayerSizes = [Dim, 96, 48], WeightLearningRate = 0.02f },
            RetrievalBeta       = 8f,
            RetrievalIterations = 5,
            BasePath            = Path.Combine(Path.GetTempPath(), "cortexapp", Guid.NewGuid().ToString("N")),
        };

        using var net = new HopfieldNetwork(cfg, rng);

        var prototypes = new[] { Cross(), Box(), Disc(), Diagonal() };

        for (int e = 0; e < epochs; e++)
            foreach (var p in prototypes)
                net.Learn(p, steps);
        foreach (var p in prototypes)
            net.Store(p, steps);

        var original  = prototypes[0];
        var corrupted = Corrupt(original, fraction: 0.35f, rng);
        var recalled  = Shape(net.Recall(corrupted, steps));
        float sim     = Cosine(original, recalled);

        try { Directory.Delete(cfg.BasePath, recursive: true); } catch { /* ignore */ }
        return new Result(Shape(original), Shape(corrupted), recalled, sim);
    }

    // ── Prototype shapes (values in [0,1]) ──────────────────────────────────────

    private static DenseTensor<float> Cross()
    {
        var t = new DenseTensor<float>(Dim);
        for (int y = 0; y < Size; y++)
        for (int x = 0; x < Size; x++)
            if (x == Size / 2 || y == Size / 2) t.Buffer[y * Size + x] = 1f;
        return t;
    }

    private static DenseTensor<float> Box()
    {
        var t = new DenseTensor<float>(Dim);
        for (int y = 0; y < Size; y++)
        for (int x = 0; x < Size; x++)
            if (x == 1 || x == Size - 2 || y == 1 || y == Size - 2) t.Buffer[y * Size + x] = 1f;
        return t;
    }

    private static DenseTensor<float> Disc()
    {
        var t = new DenseTensor<float>(Dim);
        float c = (Size - 1) / 2f, r = Size / 3f;
        for (int y = 0; y < Size; y++)
        for (int x = 0; x < Size; x++)
            if ((x - c) * (x - c) + (y - c) * (y - c) <= r * r) t.Buffer[y * Size + x] = 1f;
        return t;
    }

    private static DenseTensor<float> Diagonal()
    {
        var t = new DenseTensor<float>(Dim);
        for (int y = 0; y < Size; y++)
        for (int x = 0; x < Size; x++)
            if (Math.Abs(x - y) <= 1) t.Buffer[y * Size + x] = 1f;
        return t;
    }

    private static DenseTensor<float> Corrupt(DenseTensor<float> src, float fraction, Random rng)
    {
        var t = src.Clone();
        for (int i = 0; i < t.Length; i++)
            if (rng.NextDouble() < fraction) t.Buffer[i] = (float)rng.NextDouble();
        return t;
    }

    private static DenseTensor<float> Shape(DenseTensor<float> flat) => new(flat.Buffer, 1, Size, Size);

    private static float Cosine(DenseTensor<float> a, DenseTensor<float> b)
    {
        float dot = 0f, na = 0f, nb = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a.Buffer[i] * b.Buffer[i];
            na  += a.Buffer[i] * a.Buffer[i];
            nb  += b.Buffer[i] * b.Buffer[i];
        }
        float d = MathF.Sqrt(na) * MathF.Sqrt(nb);
        return d > 1e-6f ? dot / d : 0f;
    }
}
