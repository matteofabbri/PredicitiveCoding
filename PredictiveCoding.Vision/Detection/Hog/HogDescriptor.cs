using System.Drawing;
using PredictiveCoding.Imaging;

namespace PredictiveCoding.Vision.Detectors;

/// <summary>
/// Histogram of Oriented Gradients (Dalal &amp; Triggs, 2005) — the classic, fully
/// in-house image descriptor for a fixed detection window. Gradients are binned by
/// unsigned orientation into per-cell histograms, then 2×2 cell blocks are L2-
/// normalised and concatenated into one feature vector. Paired with a
/// <see cref="LinearModel"/> it gives a person/face detector that needs <b>no
/// external model and no download</b> at inference time.
/// </summary>
public sealed class HogDescriptor
{
    public int WindowWidth { get; }
    public int WindowHeight { get; }
    public int CellSize { get; }
    public int Bins { get; }

    private readonly int _cellsX, _cellsY, _blocksX, _blocksY;

    /// <summary>Length of the feature vector produced for a window.</summary>
    public int Dimension { get; }

    /// <param name="windowWidth">Detection window width (multiple of cellSize).</param>
    /// <param name="windowHeight">Detection window height (multiple of cellSize).</param>
    /// <param name="cellSize">Cell side in pixels (default 8).</param>
    /// <param name="bins">Unsigned orientation bins over 0–180° (default 9).</param>
    public HogDescriptor(int windowWidth = 64, int windowHeight = 128, int cellSize = 8, int bins = 9)
    {
        if (cellSize < 2) throw new ArgumentOutOfRangeException(nameof(cellSize));
        if (bins < 2) throw new ArgumentOutOfRangeException(nameof(bins));
        if (windowWidth % cellSize != 0 || windowHeight % cellSize != 0)
            throw new ArgumentException("Window dimensions must be multiples of cellSize.");

        WindowWidth = windowWidth;
        WindowHeight = windowHeight;
        CellSize = cellSize;
        Bins = bins;

        _cellsX = windowWidth / cellSize;
        _cellsY = windowHeight / cellSize;
        _blocksX = _cellsX - 1;
        _blocksY = _cellsY - 1;
        Dimension = Math.Max(0, _blocksX) * Math.Max(0, _blocksY) * (2 * 2 * bins);
        if (Dimension == 0) throw new ArgumentException("Window too small for a 2×2 block grid.");
    }

    /// <summary>Computes the HOG vector of a <see cref="WindowWidth"/>×<see cref="WindowHeight"/>
    /// grayscale buffer (row-major, values 0–255).</summary>
    public float[] ComputeGray(ReadOnlySpan<float> gray, int width, int height)
    {
        if (width != WindowWidth || height != WindowHeight)
            throw new ArgumentException($"Expected a {WindowWidth}×{WindowHeight} window, got {width}×{height}.");

        // ── Per-cell orientation histograms ──────────────────────────────────
        var cellHist = new float[_cellsX * _cellsY * Bins];
        float binWidth = 180f / Bins;

        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
            {
                float gx = gray[y * width + Clamp(x + 1, width)] - gray[y * width + Clamp(x - 1, width)];
                float gy = gray[Clamp(y + 1, height) * width + x] - gray[Clamp(y - 1, height) * width + x];

                float mag = MathF.Sqrt(gx * gx + gy * gy);
                if (mag <= 0f) continue;

                float ang = MathF.Atan2(gy, gx) * (180f / MathF.PI);
                if (ang < 0f) ang += 180f;
                if (ang >= 180f) ang -= 180f;

                int bin = Math.Min(Bins - 1, (int)(ang / binWidth));
                int cx = x / CellSize, cy = y / CellSize;
                cellHist[(cy * _cellsX + cx) * Bins + bin] += mag;
            }

        // ── 2×2 block L2 normalisation ───────────────────────────────────────
        var feature = new float[Dimension];
        int idx = 0;
        const float eps = 1e-6f;
        for (int by = 0; by < _blocksY; by++)
            for (int bx = 0; bx < _blocksX; bx++)
            {
                int start = idx;
                double sumSq = 0;
                for (int dy = 0; dy < 2; dy++)
                    for (int dx = 0; dx < 2; dx++)
                    {
                        int cell = ((by + dy) * _cellsX + (bx + dx)) * Bins;
                        for (int b = 0; b < Bins; b++)
                        {
                            float v = cellHist[cell + b];
                            feature[idx++] = v;
                            sumSq += (double)v * v;
                        }
                    }

                float norm = (float)Math.Sqrt(sumSq + eps * eps);
                for (int i = start; i < idx; i++) feature[i] /= norm;
            }

        return feature;
    }

    /// <summary>HOG of the window whose top-left is (<paramref name="x0"/>,<paramref name="y0"/>)
    /// in <paramref name="image"/> (no resizing; the window must fit).</summary>
    public float[] FromWindow(RgbImage image, int x0, int y0)
    {
        ArgumentNullException.ThrowIfNull(image);
        var gray = new float[WindowWidth * WindowHeight];
        for (int y = 0; y < WindowHeight; y++)
            for (int x = 0; x < WindowWidth; x++)
            {
                Rgb24 p = image[x0 + x, y0 + y];
                gray[y * WindowWidth + x] = 0.299f * p.R + 0.587f * p.G + 0.114f * p.B;
            }
        return ComputeGray(gray, WindowWidth, WindowHeight);
    }

    /// <summary>HOG of an arbitrary box, cropped and resized to the detection window —
    /// used to turn a teacher's detection into a training sample.</summary>
    public float[] FromCrop(RgbImage image, Rectangle box)
    {
        ArgumentNullException.ThrowIfNull(image);
        var rect = Rectangle.Intersect(box, image.Bounds);
        if (rect.Width < 1 || rect.Height < 1) return new float[Dimension];

        using RgbImage crop = image.Crop(rect).Resize(WindowWidth, WindowHeight);

        var gray = new float[WindowWidth * WindowHeight];
        for (int y = 0; y < WindowHeight; y++)
            for (int x = 0; x < WindowWidth; x++)
            {
                Rgb24 p = crop[x, y];
                gray[y * WindowWidth + x] = 0.299f * p.R + 0.587f * p.G + 0.114f * p.B;
            }
        return ComputeGray(gray, WindowWidth, WindowHeight);
    }

    private static int Clamp(int v, int n) => v < 0 ? 0 : (v >= n ? n - 1 : v);
}
