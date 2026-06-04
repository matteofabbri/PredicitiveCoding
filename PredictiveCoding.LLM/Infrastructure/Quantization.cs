using PredictiveCoding.Core;

namespace PredictiveCoding.LLM.Infrastructure;

// ─────────────────────────────────────────────────────────────────────────────
// Quantization  —  post-training weight quantization for cheaper inference.
//
//   • Int8Matrix : symmetric per-row int8 (range [−127, 127]), ~4× smaller.
//   • Int4Matrix : symmetric per-row int4 (range [−7, 7]) packed 2-per-byte, ~8×.
//
// Each row stores one fp32 scale; matvec dequantises on the fly and accumulates in
// fp32. This is standard weight-only PTQ — accuracy stays close to fp32 for int8
// and degrades gracefully for int4.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Per-row symmetric int8-quantized weight matrix.</summary>
public sealed class Int8Matrix
{
    private readonly sbyte[] _q;        // [rows*cols]
    private readonly float[] _scale;    // [rows]

    public int Rows { get; }
    public int Cols { get; }

    /// <summary>Bytes of storage (quantized weights + per-row scales).</summary>
    public long MemoryBytes => (long)_q.Length + (long)_scale.Length * sizeof(float);

    private Int8Matrix(int rows, int cols, sbyte[] q, float[] scale)
    {
        Rows = rows; Cols = cols; _q = q; _scale = scale;
    }

    public static Int8Matrix FromFloat(DenseTensor<float> w)
    {
        if (w.Rank != 2) throw new ArgumentException("weight must be [rows, cols].", nameof(w));
        int rows = w.Dimensions[0], cols = w.Dimensions[1];
        var q = new sbyte[rows * cols];
        var scale = new float[rows];

        for (int r = 0; r < rows; r++)
        {
            int baseIdx = r * cols;
            float max = 1e-12f;
            for (int c = 0; c < cols; c++) max = MathF.Max(max, MathF.Abs(w.Buffer[baseIdx + c]));
            float s = max / 127f;
            scale[r] = s;
            float inv = 1f / s;
            for (int c = 0; c < cols; c++)
                q[baseIdx + c] = (sbyte)Math.Clamp((int)MathF.Round(w.Buffer[baseIdx + c] * inv), -127, 127);
        }
        return new Int8Matrix(rows, cols, q, scale);
    }

    public DenseTensor<float> MatVec(DenseTensor<float> x)
    {
        if (x.Length != Cols) throw new ArgumentException($"x has {x.Length} elements; expected {Cols}.");
        var y = new DenseTensor<float>(Rows);
        for (int r = 0; r < Rows; r++)
        {
            int baseIdx = r * Cols;
            float acc = 0f;
            for (int c = 0; c < Cols; c++) acc += _q[baseIdx + c] * x.Buffer[c];
            y.Buffer[r] = acc * _scale[r];           // dequantize after integer-weighted accumulation
        }
        return y;
    }
}

/// <summary>Per-row symmetric int4-quantized weight matrix, packed two values per byte.</summary>
public sealed class Int4Matrix
{
    private readonly byte[]  _packed;   // [rows * ceil(cols/2)]
    private readonly float[] _scale;    // [rows]

    public int Rows { get; }
    public int Cols { get; }

    public long MemoryBytes => (long)_packed.Length + (long)_scale.Length * sizeof(float);

    private Int4Matrix(int rows, int cols, byte[] packed, float[] scale)
    {
        Rows = rows; Cols = cols; _packed = packed; _scale = scale;
    }

    private static int RowStride(int cols) => (cols + 1) / 2;

    public static Int4Matrix FromFloat(DenseTensor<float> w)
    {
        if (w.Rank != 2) throw new ArgumentException("weight must be [rows, cols].", nameof(w));
        int rows = w.Dimensions[0], cols = w.Dimensions[1];
        int stride = RowStride(cols);
        var packed = new byte[rows * stride];
        var scale  = new float[rows];

        for (int r = 0; r < rows; r++)
        {
            int wBase = r * cols, pBase = r * stride;
            float max = 1e-12f;
            for (int c = 0; c < cols; c++) max = MathF.Max(max, MathF.Abs(w.Buffer[wBase + c]));
            float s = max / 7f;                       // int4 symmetric range [-7, 7]
            scale[r] = s;
            float inv = 1f / s;
            for (int c = 0; c < cols; c++)
            {
                int q = Math.Clamp((int)MathF.Round(w.Buffer[wBase + c] * inv), -7, 7);
                byte nibble = (byte)(q & 0x0F);       // two's-complement low nibble
                int  pi = pBase + (c >> 1);
                if ((c & 1) == 0) packed[pi] = nibble; // low nibble
                else              packed[pi] |= (byte)(nibble << 4);
            }
        }
        return new Int4Matrix(rows, cols, packed, scale);
    }

    public DenseTensor<float> MatVec(DenseTensor<float> x)
    {
        if (x.Length != Cols) throw new ArgumentException($"x has {x.Length} elements; expected {Cols}.");
        int stride = RowStride(Cols);
        var y = new DenseTensor<float>(Rows);
        for (int r = 0; r < Rows; r++)
        {
            int pBase = r * stride;
            float acc = 0f;
            for (int c = 0; c < Cols; c++)
            {
                byte b      = _packed[pBase + (c >> 1)];
                int  nibble = ((c & 1) == 0) ? (b & 0x0F) : (b >> 4) & 0x0F;
                int  q      = nibble >= 8 ? nibble - 16 : nibble;   // sign-extend 4-bit
                acc += q * x.Buffer[c];
            }
            y.Buffer[r] = acc * _scale[r];
        }
        return y;
    }
}
