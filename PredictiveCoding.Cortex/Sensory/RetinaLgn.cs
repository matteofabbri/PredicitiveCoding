using PredictiveCoding.Core;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// RetinaLgn  —  retinal / LGN front-end (center-surround predictive coding)
//
// The earliest stage of Rao & Ballard's (1999) predictive-coding account: retinal
// ganglion / LGN cells have center-surround receptive fields that subtract a
// local prediction (the surround) from each point (the center), whitening natural
// images and passing on only the unpredicted residual.  Implemented as a
// Difference-of-Gaussians (DoG) spatial filter per channel — a fixed, biologically
// motivated whitening transform that feeds the cortical hierarchy.
//
// References: Rao & Ballard (1999); Atick & Redlich (1992) retinal whitening.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class RetinaLgn
{
    public int   Channels     { get; }
    public int   Height       { get; }
    public int   Width        { get; }
    public float CenterSigma  { get; }
    public float SurroundSigma { get; }

    public RetinaLgn(int channels, int height, int width,
                     float centerSigma = 0.6f, float surroundSigma = 1.5f)
    {
        if (surroundSigma <= centerSigma)
            throw new ArgumentException("surroundSigma must exceed centerSigma (center-surround antagonism).");
        Channels = channels; Height = height; Width = width;
        CenterSigma = centerSigma; SurroundSigma = surroundSigma;
    }

    /// <summary>
    /// Applies the center-surround (DoG) transform to a [C,H,W] image and returns
    /// the whitened residual of the same shape: center blur − surround blur.
    /// </summary>
    public DenseTensor<float> Process(DenseTensor<float> image)
    {
        if (image.Length != Channels * Height * Width)
            throw new ArgumentException(
                $"image has {image.Length} elements; expected {Channels * Height * Width} (C·H·W).");

        var center   = GaussianBlur(image, CenterSigma);
        var surround = GaussianBlur(image, SurroundSigma);
        var outp     = new DenseTensor<float>(Channels, Height, Width);
        for (int i = 0; i < outp.Length; i++)
            outp.Buffer[i] = center.Buffer[i] - surround.Buffer[i];
        return outp;
    }

    // ── Separable Gaussian blur (per channel, clamped edges) ────────────────────

    private DenseTensor<float> GaussianBlur(DenseTensor<float> image, float sigma)
    {
        int   radius = Math.Max(1, (int)MathF.Ceiling(3f * sigma));
        var   kernel = new float[2 * radius + 1];
        float sum = 0f;
        for (int k = -radius; k <= radius; k++)
        {
            float w = MathF.Exp(-(k * k) / (2f * sigma * sigma));
            kernel[k + radius] = w;
            sum += w;
        }
        for (int k = 0; k < kernel.Length; k++) kernel[k] /= sum;

        var tmp = new DenseTensor<float>(Channels, Height, Width);
        var dst = new DenseTensor<float>(Channels, Height, Width);

        // Horizontal pass: image → tmp
        for (int c = 0; c < Channels; c++)
        for (int y = 0; y < Height; y++)
        for (int x = 0; x < Width;  x++)
        {
            float acc = 0f;
            for (int k = -radius; k <= radius; k++)
            {
                int xx = Math.Clamp(x + k, 0, Width - 1);
                acc += kernel[k + radius] * image[c, y, xx];
            }
            tmp[c, y, x] = acc;
        }

        // Vertical pass: tmp → dst
        for (int c = 0; c < Channels; c++)
        for (int y = 0; y < Height; y++)
        for (int x = 0; x < Width;  x++)
        {
            float acc = 0f;
            for (int k = -radius; k <= radius; k++)
            {
                int yy = Math.Clamp(y + k, 0, Height - 1);
                acc += kernel[k + radius] * tmp[c, yy, x];
            }
            dst[c, y, x] = acc;
        }

        return dst;
    }
}
