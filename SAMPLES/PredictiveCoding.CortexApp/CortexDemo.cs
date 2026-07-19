using PredictiveCoding.Core;
using PredictiveCoding.Cortex;
using PredictiveCoding.Networks.Hierarchical;
using PredictiveCoding.Networks;

namespace PredictiveCoding.CortexApp;

// ─────────────────────────────────────────────────────────────────────────────
// CortexDemo  —  drives a VisualCortex on a tiny synthetic image set.
//
// Four visual "object classes" (each a distinct low-level structure) are
// generated procedurally.  A hierarchical predictive-coding visual cortex is
// trained on noisy instances, then a fresh noisy test image is classified and
// reconstructed top-down — demonstrating perception (recognition) and
// generation (reconstruction) in the same model.
// ─────────────────────────────────────────────────────────────────────────────

public static class CortexDemo
{
    public const int Channels = 3;
    public const int Size     = 24;             // H = W = 24

    public static readonly string[] ClassNames =
        ["H-gradient", "V-gradient", "Checker", "Disc"];

    public sealed record Result(
        DenseTensor<float> Input,
        DenseTensor<float> Reconstruction,
        int                TrueClass,
        int                PredClass,
        float              ReconError,
        float              V1Energy,
        float[]            AreaEnergies);

    /// <summary>
    /// Builds, trains and evaluates a VisualCortex.  Runs synchronously, so call
    /// it from a background thread; <paramref name="progress"/> is invoked with
    /// human-readable status updates.
    /// </summary>
    public static Result Run(int epochs, int inferenceSteps, Action<string>? progress = null)
    {
        int classes = ClassNames.Length;
        var rng     = new Random(1234);

        var config = new VisualCortexConfig
        {
            ImageChannels = Channels,
            ImageHeight   = Size,
            ImageWidth    = Size,
            PatchSize     = 8,
            PatchStride   = 8,                                   // 3×3 grid of columns
            V1Column      = new NetworkConfig { LayerSizes = [Channels * 8 * 8, 64, 16] },
            HigherAreas   = new HierarchicalNetworkConfig
            {
                Modules =
                [
                    new NetworkConfig { LayerSizes = [3 * 3 * 16, 64] }, // V2
                    new NetworkConfig { LayerSizes = [64, 32] },         // V4 → IT
                ],
            },
            Classifier = new NetworkConfig { LayerSizes = [32, 4] },     // IT → class
            BasePath   = Path.Combine(Path.GetTempPath(), "cortexapp", Guid.NewGuid().ToString("N")),
        };

        using var cortex = new VisualCortex(config, rng);

        // ── Training ──────────────────────────────────────────────────────────
        for (int e = 0; e < epochs; e++)
        {
            for (int c = 0; c < classes; c++)
            {
                var img   = GenerateClassImage(c, noise: 0.08f, rng);
                var label = OneHot(c, classes);
                cortex.LearnSupervised(img, label, inferenceSteps);
            }
            progress?.Invoke($"Training… epoch {e + 1}/{epochs}");
        }

        // ── Evaluation ────────────────────────────────────────────────────────
        progress?.Invoke("Evaluating…");
        int trueClass = rng.Next(classes);
        var test      = GenerateClassImage(trueClass, noise: 0.15f, rng);

        int predClass = cortex.Predict(test, inferenceSteps);
        var recon     = cortex.Reconstruct(test, inferenceSteps);
        float err     = cortex.ReconstructionError(test, inferenceSteps);

        float v1   = cortex.V1Energy();
        var areas  = new float[config.HigherAreas.Modules.Length];
        for (int a = 0; a < areas.Length; a++) areas[a] = cortex.HigherAreaEnergy(a);

        // Best-effort cleanup of the FasterKV store directory.
        try { Directory.Delete(config.BasePath, recursive: true); } catch { /* ignore */ }

        return new Result(test, recon, trueClass, predClass, err, v1, areas);
    }

    // ── Synthetic image generation (values in [0, 1]) ───────────────────────────

    public static DenseTensor<float> GenerateClassImage(int cls, float noise, Random rng)
    {
        var img = new DenseTensor<float>(Channels, Size, Size);

        for (int y = 0; y < Size; y++)
        for (int x = 0; x < Size; x++)
        {
            float r = 0f, g = 0f, b = 0f;
            switch (cls)
            {
                case 0: r = (float)x / (Size - 1); break;                  // horizontal gradient (red)
                case 1: g = (float)y / (Size - 1); break;                  // vertical gradient (green)
                case 2: b = ((x / 4 + y / 4) % 2 == 0) ? 1f : 0.1f; break; // checkerboard (blue)
                default:                                                    // centred disc (white)
                    float dx = x - (Size - 1) / 2f, dy = y - (Size - 1) / 2f;
                    float v = MathF.Sqrt(dx * dx + dy * dy) < Size / 3f ? 1f : 0f;
                    r = g = b = v;
                    break;
            }
            img[0, y, x] = Clamp01(r + Noise(noise, rng));
            img[1, y, x] = Clamp01(g + Noise(noise, rng));
            img[2, y, x] = Clamp01(b + Noise(noise, rng));
        }
        return img;
    }

    private static float Noise(float sigma, Random rng) =>
        sigma <= 0f ? 0f : RandomUtils.SampleNormal(0f, sigma, rng);

    private static float Clamp01(float v) => v < 0f ? 0f : v > 1f ? 1f : v;

    private static DenseTensor<float> OneHot(int index, int count)
    {
        var t = DenseTensor<float>.Zeros(count);
        t.Buffer[index] = 1f;
        return t;
    }
}
