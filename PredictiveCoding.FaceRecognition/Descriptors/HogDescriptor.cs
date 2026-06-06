namespace PredictiveCoding.FaceRecognition.Descriptors;

/// <summary>
/// Standard <b>Histogram of Oriented Gradients (HOG)</b> descriptor for face recognition.
/// Layout (for a 128 × 128 image):
///   • 8 × 8 non-overlapping cells of 16 × 16 pixels each.
///   • 8 signed orientation bins per cell (0°–360°, bin width = 45°).
///   • Overlapping 2 × 2 cell blocks with stride 1, L2-normalised.
///   • (8−1)² blocks × 4 cells × 8 bins = <b>1568-D</b> feature vector.
/// HOG is complementary to LBP: LBP captures texture (binary patterns), HOG
/// captures gradient structure (edge shape and orientation).
/// </summary>
public static class HogDescriptor
{
    public const int CellsPerAxis = 8;   // 8 cells per spatial axis
    public const int NumBins      = 8;   // signed orientation bins (0–360°)
    public const int BlockCells   = 2;   // 2 × 2 cells per block

    // Number of overlapping blocks per axis (stride = 1 cell)
    private const int NumBlocks = CellsPerAxis - BlockCells + 1; // 7

    /// <summary>Total descriptor dimensionality: 7 × 7 × 4 × 8 = 1568.</summary>
    public const int Dimension = NumBlocks * NumBlocks * BlockCells * BlockCells * NumBins;

    /// <summary>Computes the HOG descriptor from pre-loaded grayscale pixels of <paramref name="size"/> × <paramref name="size"/>.</summary>
    public static float[] Compute(byte[] gray, int size = 128)
    {
        int cellPx = size / CellsPerAxis;  // pixels per cell side

        // ── Cell histograms ───────────────────────────────────────────────────────
        // hist[cy, cx, b] = magnitude accumulation for cell (cy,cx), bin b
        var hist = new float[CellsPerAxis * CellsPerAxis * NumBins];

        for (int y = 1; y < size - 1; y++)
        {
            for (int x = 1; x < size - 1; x++)
            {
                float gx = gray[y * size + x + 1] - gray[y * size + x - 1];
                float gy = gray[(y + 1) * size + x] - gray[(y - 1) * size + x];
                float mag   = MathF.Sqrt(gx * gx + gy * gy);
                float angle = MathF.Atan2(gy, gx) * (180f / MathF.PI);
                if (angle < 0) angle += 360f;   // [0, 360)

                // Bilinear interpolation between two adjacent bins
                float binF = angle / 360f * NumBins;
                int   b0   = (int)binF % NumBins;
                int   b1   = (b0 + 1) % NumBins;
                float frac = binF - MathF.Floor(binF);

                int cy = Math.Min(y / cellPx, CellsPerAxis - 1);
                int cx = Math.Min(x / cellPx, CellsPerAxis - 1);
                int ci = (cy * CellsPerAxis + cx) * NumBins;

                hist[ci + b0] += mag * (1f - frac);
                hist[ci + b1] += mag * frac;
            }
        }

        // ── Block normalisation (overlapping 2 × 2, stride 1, L2) ─────────────
        var result = new float[Dimension];
        int outIdx = 0;

        for (int by = 0; by < NumBlocks; by++)
        {
            for (int bx = 0; bx < NumBlocks; bx++)
            {
                // Gather block vector + compute L2 norm
                float norm = 1e-6f;
                for (int cy = by; cy < by + BlockCells; cy++)
                for (int cx = bx; cx < bx + BlockCells; cx++)
                {
                    int ci = (cy * CellsPerAxis + cx) * NumBins;
                    for (int b = 0; b < NumBins; b++)
                        norm += hist[ci + b] * hist[ci + b];
                }
                norm = MathF.Sqrt(norm);

                // Write normalised block
                for (int cy = by; cy < by + BlockCells; cy++)
                for (int cx = bx; cx < bx + BlockCells; cx++)
                {
                    int ci = (cy * CellsPerAxis + cx) * NumBins;
                    for (int b = 0; b < NumBins; b++)
                        result[outIdx++] = hist[ci + b] / norm;
                }
            }
        }

        return result;
    }
}
