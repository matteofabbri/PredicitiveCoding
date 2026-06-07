using PredictiveCoding.Vision.Detectors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PredictiveCoding.Vision.ReId;

/// <summary>
/// A dependency-free appearance embedder: the per-channel RGB colour histogram of
/// the box's central region, L2-normalised. People in differently-coloured clothing
/// separate well, it needs no model or GPU, and it is a sensible re-ID baseline that
/// can later be swapped for a face/body deep embedder via <see cref="IAppearanceEmbedder"/>.
/// Only the central fraction of the box is sampled, which trims away most of the
/// background around the person.
/// </summary>
public sealed class ColorHistogramEmbedder : IAppearanceEmbedder
{
    private readonly int _bins;
    private readonly float _centralCrop;

    public string Name => "color-histogram";
    public int Dimension => _bins * 3;

    /// <param name="binsPerChannel">Histogram bins per RGB channel (1–256). Default 8.</param>
    /// <param name="centralCrop">Fraction of the box (centred) to sample, in (0,1]. Default 0.6.</param>
    public ColorHistogramEmbedder(int binsPerChannel = 8, float centralCrop = 0.6f)
    {
        if (binsPerChannel is < 1 or > 256)
            throw new ArgumentOutOfRangeException(nameof(binsPerChannel));
        if (centralCrop is <= 0f or > 1f)
            throw new ArgumentOutOfRangeException(nameof(centralCrop));
        _bins = binsPerChannel;
        _centralCrop = centralCrop;
    }

    public float[] Embed(Image<Rgb24> frame, Detection box)
    {
        ArgumentNullException.ThrowIfNull(frame);
        var hist = new float[_bins * 3];

        float marginX = box.Width * (1f - _centralCrop) * 0.5f;
        float marginY = box.Height * (1f - _centralCrop) * 0.5f;
        int x0 = Math.Clamp((int)MathF.Round(box.X1 + marginX), 0, frame.Width - 1);
        int y0 = Math.Clamp((int)MathF.Round(box.Y1 + marginY), 0, frame.Height - 1);
        int x1 = Math.Clamp((int)MathF.Round(box.X2 - marginX), 0, frame.Width - 1);
        int y1 = Math.Clamp((int)MathF.Round(box.Y2 - marginY), 0, frame.Height - 1);

        int count = 0;
        for (int y = y0; y <= y1; y++)
            for (int x = x0; x <= x1; x++)
            {
                Rgb24 p = frame[x, y];
                hist[Bin(p.R)]++;
                hist[_bins + Bin(p.G)]++;
                hist[2 * _bins + Bin(p.B)]++;
                count++;
            }

        if (count == 0) return hist;   // empty region → all-zero vector

        double norm = 0;
        foreach (float v in hist) norm += (double)v * v;
        norm = Math.Sqrt(norm);
        if (norm > 1e-8)
            for (int i = 0; i < hist.Length; i++) hist[i] = (float)(hist[i] / norm);
        return hist;
    }

    private int Bin(byte value) => Math.Min(_bins - 1, value * _bins / 256);
}
