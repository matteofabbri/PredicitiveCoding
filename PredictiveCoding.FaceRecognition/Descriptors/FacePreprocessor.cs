namespace PredictiveCoding.FaceRecognition.Descriptors;

/// <summary>
/// Illumination preprocessing pipeline for face images.
/// Implements the <b>Tan &amp; Triggs (2010)</b> method:
///   1. Gamma correction (γ = 0.2) — compresses the dynamic range.
///   2. Difference of Gaussians (DoG) — removes slow-varying illumination gradients.
///   3. Robust contrast equalization (α = 0.1) — global scale invariance.
///   4. Tanh squashing (τ = 10) — suppresses remaining outliers.
/// The output is rescaled to [0, 255] so it can be passed directly to
/// <see cref="FaceDescriptor.ComputeFromPixels"/> or <see cref="HogDescriptor.Compute"/>.
/// </summary>
public static class FacePreprocessor
{
    private const float Gamma = 0.2f;
    private const float Alpha = 0.1f;   // contrast equalization exponent
    private const float Tau   = 10.0f;  // tanh saturation scale

    // DoG radii (box-blur approximation to Gaussian).
    // radius=2 ≈ σ≈1.15 px (fine scale);  radius=4 ≈ σ≈2.31 px (coarse scale)
    private const int FineRadius   = 2;
    private const int CoarseRadius = 4;

    /// <summary>
    /// Applies the full Tan &amp; Triggs normalization and returns [0, 255] grayscale bytes.
    /// </summary>
    public static byte[] NormalizeTanTriggs(byte[] gray, int size)
    {
        int n = size * size;
        var img = new float[n];

        // 1. Gamma correction: work in raw-pixel space (not normalized to [0,1])
        //    so subsequent DoG values stay in a useful numeric range.
        for (int i = 0; i < n; i++)
            img[i] = MathF.Pow(gray[i] + 0.5f, Gamma);  // +0.5 avoids log(0)

        // 2. DoG = fine_blur – coarse_blur
        var fine   = BoxBlur(img, size, FineRadius);
        var coarse = BoxBlur(img, size, CoarseRadius);
        for (int i = 0; i < n; i++)
            img[i] = fine[i] - coarse[i];

        // 3. Robust contrast equalization: divide by (mean |I|^α)^{1/α}
        //    This is the α-power mean — gives robustness against outlier pixels.
        float powMean = 0f;
        for (int i = 0; i < n; i++)
            powMean += MathF.Pow(MathF.Abs(img[i]) + 1e-6f, Alpha);
        powMean /= n;
        float scale = MathF.Pow(powMean, 1f / Alpha);
        if (scale > 1e-8f)
            for (int i = 0; i < n; i++)
                img[i] /= scale;

        // 4. Tanh squashing: I ← τ · tanh(I/τ)
        for (int i = 0; i < n; i++)
            img[i] = Tau * MathF.Tanh(img[i] / Tau);

        // 5. Rescale to [0, 255] (LBP only needs relative ordering; tanh is monotone).
        float min = img[0], max = img[0];
        for (int i = 1; i < n; i++) { if (img[i] < min) min = img[i]; if (img[i] > max) max = img[i]; }
        float range = max - min;

        var result = new byte[n];
        for (int i = 0; i < n; i++)
            result[i] = range > 1e-8f
                ? (byte)MathF.Round((img[i] - min) / range * 255f)
                : (byte)128;
        return result;
    }

    /// <summary>
    /// Applies gamma correction only (γ = 0.2), rescaled to [0, 255].
    /// Lighter-weight alternative used as input to HOG where DoG can hurt gradients.
    /// </summary>
    public static byte[] GammaCorrect(byte[] gray, int size)
    {
        int n = size * size;
        var result = new byte[n];
        for (int i = 0; i < n; i++)
            result[i] = (byte)MathF.Round(MathF.Pow((gray[i] + 0.5f) / 256f, Gamma) * 255f);
        return result;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────────

    /// <summary>Separable box blur with border clamping.</summary>
    private static float[] BoxBlur(float[] input, int size, int radius)
    {
        var tmp    = new float[input.Length];
        var output = new float[input.Length];

        // Horizontal pass
        for (int y = 0; y < size; y++)
        {
            for (int x = 0; x < size; x++)
            {
                float sum = 0f;
                int   cnt = 0;
                for (int dx = -radius; dx <= radius; dx++)
                {
                    sum += input[y * size + Math.Clamp(x + dx, 0, size - 1)];
                    cnt++;
                }
                tmp[y * size + x] = sum / cnt;
            }
        }

        // Vertical pass
        for (int x = 0; x < size; x++)
        {
            for (int y = 0; y < size; y++)
            {
                float sum = 0f;
                int   cnt = 0;
                for (int dy = -radius; dy <= radius; dy++)
                {
                    sum += tmp[Math.Clamp(y + dy, 0, size - 1) * size + x];
                    cnt++;
                }
                output[y * size + x] = sum / cnt;
            }
        }

        return output;
    }
}
