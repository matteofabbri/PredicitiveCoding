using PredictiveCoding.Core;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// RetinotopicMap  —  receptive-field tiling of an image (the V1 "retinotopy")
//
// The visual cortex preserves the spatial layout of the retina: nearby points in
// the visual field map to nearby cortical columns, and each V1 column sees only a
// small local patch (its receptive field).  This helper turns a channels-first
// image [C, H, W] into a grid of overlapping P×P receptive-field patches and back
// again, so a single weight-shared cortical column can be applied at every
// location (convolutional weight sharing).
//
// Patch layout matches the image: a patch is a flat [C·P·P] vector indexed
//   c·P·P + dy·P + dx  ←  image[c, py+dy, px+dx].
//
// Reassembly averages overlapping patches, which is exactly the consensus a
// predictive-coding generative model produces when several columns predict the
// same pixels.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class RetinotopicMap
{
    public int Channels  { get; }
    public int Height    { get; }
    public int Width     { get; }
    public int PatchSize { get; }
    public int Stride    { get; }

    /// <summary>Number of receptive fields along the vertical axis.</summary>
    public int GridHeight { get; }

    /// <summary>Number of receptive fields along the horizontal axis.</summary>
    public int GridWidth  { get; }

    /// <summary>Total receptive fields (cortical columns) = GridHeight · GridWidth.</summary>
    public int PatchCount => GridHeight * GridWidth;

    /// <summary>Flattened size of one receptive-field patch = Channels · PatchSize².</summary>
    public int PatchDim => Channels * PatchSize * PatchSize;

    public RetinotopicMap(int channels, int height, int width, int patchSize, int stride)
    {
        if (channels < 1) throw new ArgumentOutOfRangeException(nameof(channels));
        if (height < 1)   throw new ArgumentOutOfRangeException(nameof(height));
        if (width < 1)    throw new ArgumentOutOfRangeException(nameof(width));
        if (patchSize < 1 || patchSize > height || patchSize > width)
            throw new ArgumentOutOfRangeException(nameof(patchSize),
                "patchSize must be in [1, min(height, width)].");
        if (stride < 1) throw new ArgumentOutOfRangeException(nameof(stride));

        Channels  = channels;
        Height    = height;
        Width     = width;
        PatchSize = patchSize;
        Stride    = stride;

        GridHeight = GridSize(height, patchSize, stride);
        GridWidth  = GridSize(width,  patchSize, stride);
    }

    /// <summary>Number of patch positions that fit along a dimension of size <paramref name="dim"/>.</summary>
    public static int GridSize(int dim, int patch, int stride) => (dim - patch) / stride + 1;

    // ── Extraction ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Splits <paramref name="image"/> (flat or [C,H,W]) into <see cref="PatchCount"/>
    /// receptive-field patches, ordered row-major over the grid (gy, gx).
    /// Each returned tensor is a rank-1 vector of length <see cref="PatchDim"/>.
    /// </summary>
    public DenseTensor<float>[] Extract(DenseTensor<float> image)
    {
        if (image.Length != Channels * Height * Width)
            throw new ArgumentException(
                $"image has {image.Length} elements; expected {Channels * Height * Width} (C·H·W).");

        var src     = image.Buffer;
        var patches = new DenseTensor<float>[PatchCount];
        int p       = 0;

        for (int gy = 0; gy < GridHeight; gy++)
        for (int gx = 0; gx < GridWidth;  gx++)
        {
            int py = gy * Stride;
            int px = gx * Stride;
            var patch = new DenseTensor<float>(PatchDim);
            var dst   = patch.Buffer;

            for (int c = 0; c < Channels; c++)
            {
                int chBase  = c * Height * Width;
                int outBase = c * PatchSize * PatchSize;
                for (int dy = 0; dy < PatchSize; dy++)
                {
                    int srcRow = chBase + (py + dy) * Width + px;
                    int dstRow = outBase + dy * PatchSize;
                    Array.Copy(src, srcRow, dst, dstRow, PatchSize);
                }
            }
            patches[p++] = patch;
        }
        return patches;
    }

    // ── Reassembly ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Reassembles a full [C,H,W] image from per-patch reconstructions, averaging
    /// pixels covered by more than one receptive field.  <paramref name="patches"/>
    /// must be in the same grid order produced by <see cref="Extract"/>.
    /// </summary>
    public DenseTensor<float> Assemble(IReadOnlyList<DenseTensor<float>> patches)
    {
        if (patches.Count != PatchCount)
            throw new ArgumentException($"expected {PatchCount} patches but got {patches.Count}.");

        var image = new DenseTensor<float>(Channels, Height, Width);
        var acc   = image.Buffer;
        var count = new float[acc.Length];
        int p     = 0;

        for (int gy = 0; gy < GridHeight; gy++)
        for (int gx = 0; gx < GridWidth;  gx++)
        {
            int py  = gy * Stride;
            int px  = gx * Stride;
            var pv  = patches[p++];
            if (pv.Length != PatchDim)
                throw new ArgumentException($"patch has {pv.Length} elements; expected {PatchDim}.");

            for (int c = 0; c < Channels; c++)
            {
                int chBase = c * Height * Width;
                int inBase = c * PatchSize * PatchSize;
                for (int dy = 0; dy < PatchSize; dy++)
                for (int dx = 0; dx < PatchSize; dx++)
                {
                    int img = chBase + (py + dy) * Width + (px + dx);
                    acc[img]   += pv.Buffer[inBase + dy * PatchSize + dx];
                    count[img] += 1f;
                }
            }
        }

        for (int i = 0; i < acc.Length; i++)
            if (count[i] > 0f) acc[i] /= count[i];

        return image;
    }
}
