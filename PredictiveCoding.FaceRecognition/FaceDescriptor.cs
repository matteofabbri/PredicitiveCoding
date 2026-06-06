using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PredictiveCoding.FaceRecognition;

/// <summary>
/// Turns a face image into a fixed-length, illumination-tolerant descriptor: a grid of <b>uniform Local Binary
/// Pattern</b> histograms (the classic strong hand-crafted LFW feature) over a centre-cropped, grayscale face.
/// The centre crop is a cheap stand-in for face alignment — it drops the background/hair that dominate raw
/// images. This is the input representation a learnable embedder (e.g. the Predictive Coding one) sits on top of.
/// </summary>
public static class FaceDescriptor
{
    private const int Size = 128;            // working image after crop+resize
    private const int Grid = 8;              // Size/Grid cells per axis
    private const int Bins = 59;             // 58 uniform LBP patterns + 1 non-uniform
    private const float CropFraction = 0.62f; // central square kept (alignment proxy)
    public const int Dimension = Grid * Grid * Bins;

    private static readonly int[] UniformMap = BuildUniformMap();

    /// <summary>Full pipeline: load → crop → resize → LBP descriptor.</summary>
    public static float[] Compute(string imagePath)
        => ComputeFromPixels(LoadPixels(imagePath));

    /// <summary>
    /// Loads, centre-crops and resizes the image to <c>Size × Size</c> grayscale.
    /// Call this once and share the pixels between LBP, HOG, and FacePreprocessor
    /// so each face image is only decoded once.
    /// </summary>
    public static byte[] LoadPixels(string imagePath)
    {
        using Image<L8> image = Image.Load<L8>(imagePath);
        int side = Math.Max(1, (int)(Math.Min(image.Width, image.Height) * CropFraction));
        int x0 = (image.Width - side) / 2;
        int y0 = (image.Height - side) / 2;
        image.Mutate(ctx => ctx
            .Crop(new Rectangle(x0, y0, side, side))
            .Resize(Size, Size));

        var gray = new byte[Size * Size];
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < Size; y++)
            {
                Span<L8> row = accessor.GetRowSpan(y);
                for (int x = 0; x < Size; x++)
                    gray[(y * Size) + x] = row[x].PackedValue;
            }
        });
        return gray;
    }

    /// <summary>
    /// Computes the LBP descriptor from pre-loaded grayscale pixels of size <c>Size × Size</c>.
    /// Lets callers share image loading with HOG and other descriptors.
    /// </summary>
    public static float[] ComputeFromPixels(byte[] gray)
    {
        int cell = Size / Grid;
        var feature = new float[Dimension];
        for (int y = 1; y < Size - 1; y++)
        {
            for (int x = 1; x < Size - 1; x++)
            {
                byte c = gray[(y * Size) + x];
                int code =
                    (Ge(gray, y - 1, x - 1, c) << 0) |
                    (Ge(gray, y - 1, x,     c) << 1) |
                    (Ge(gray, y - 1, x + 1, c) << 2) |
                    (Ge(gray, y,     x + 1, c) << 3) |
                    (Ge(gray, y + 1, x + 1, c) << 4) |
                    (Ge(gray, y + 1, x,     c) << 5) |
                    (Ge(gray, y + 1, x - 1, c) << 6) |
                    (Ge(gray, y,     x - 1, c) << 7);

                int cellIndex = (Math.Min(y / cell, Grid - 1) * Grid) + Math.Min(x / cell, Grid - 1);
                feature[(cellIndex * Bins) + UniformMap[code]]++;
            }
        }

        // Per-cell L1 + signed-sqrt power normalisation, then global L2.
        for (int cellIndex = 0; cellIndex < Grid * Grid; cellIndex++)
        {
            int off = cellIndex * Bins;
            double sum = 0;
            for (int i = 0; i < Bins; i++)
                sum += feature[off + i];
            if (sum > 0)
                for (int i = 0; i < Bins; i++)
                    feature[off + i] = (float)Math.Sqrt(feature[off + i] / sum);
        }

        double norm = 0;
        foreach (float f in feature)
            norm += (double)f * f;
        norm = Math.Sqrt(norm);
        if (norm > 0)
            for (int i = 0; i < feature.Length; i++)
                feature[i] = (float)(feature[i] / norm);

        return feature;
    }

    private static int Ge(byte[] gray, int y, int x, byte center)
        => gray[(y * Size) + x] >= center ? 1 : 0;

    private static int[] BuildUniformMap()
    {
        var map = new int[256];
        int next = 0;
        for (int code = 0; code < 256; code++)
            map[code] = Transitions(code) <= 2 ? next++ : 58;
        return map; // next == 58 uniform patterns -> bins 0..57, non-uniform -> 58
    }

    private static int Transitions(int code)
    {
        int t = 0;
        for (int i = 0; i < 8; i++)
        {
            int b1 = (code >> i) & 1;
            int b2 = (code >> ((i + 1) % 8)) & 1;
            if (b1 != b2)
                t++;
        }
        return t;
    }
}
