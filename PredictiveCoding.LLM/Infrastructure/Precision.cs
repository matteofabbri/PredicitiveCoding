using PredictiveCoding.Core;

namespace PredictiveCoding.LLM.Infrastructure;

// ─────────────────────────────────────────────────────────────────────────────
// Precision  —  mixed-precision weight storage (fp16 / bf16).
//
// Weights are stored at 16 bits (halving memory and bandwidth) and up-cast to
// fp32 for the actual dot products, exactly like a mixed-precision inference path
// (store low, accumulate high). fp16 uses System.Half; bf16 uses the BFloat16
// helper (truncate the fp32 mantissa, round to nearest even).
// ─────────────────────────────────────────────────────────────────────────────

public enum LowPrecision { Fp16, Bf16 }

/// <summary>Brain-floating-point (bfloat16): the top 16 bits of an fp32 value.</summary>
public readonly struct BFloat16
{
    public readonly ushort Bits;
    public BFloat16(ushort bits) => Bits = bits;

    public static BFloat16 FromFloat(float f)
    {
        uint bits = BitConverter.SingleToUInt32Bits(f);
        if (float.IsNaN(f)) return new BFloat16((ushort)((bits >> 16) | 0x0040)); // keep it NaN
        uint rounding = 0x7FFFu + ((bits >> 16) & 1u);                            // round to nearest even
        return new BFloat16((ushort)((bits + rounding) >> 16));
    }

    public float ToFloat() => BitConverter.UInt32BitsToSingle((uint)Bits << 16);
}

/// <summary>A weight matrix stored in 16-bit precision; matvec accumulates in fp32.</summary>
public sealed class LowPrecisionMatrix
{
    private readonly ushort[]     _data;     // raw 16-bit codes (Half bits or bf16 bits)
    private readonly LowPrecision _prec;

    public int Rows { get; }
    public int Cols { get; }

    /// <summary>Bytes of weight storage (≈ half of the fp32 original).</summary>
    public long MemoryBytes => (long)_data.Length * sizeof(ushort);

    private LowPrecisionMatrix(int rows, int cols, ushort[] data, LowPrecision prec)
    {
        Rows = rows; Cols = cols; _data = data; _prec = prec;
    }

    public static LowPrecisionMatrix FromFloat(DenseTensor<float> w, LowPrecision precision)
    {
        if (w.Rank != 2) throw new ArgumentException("weight must be a [rows, cols] matrix.", nameof(w));
        int rows = w.Dimensions[0], cols = w.Dimensions[1];
        var data = new ushort[rows * cols];
        for (int i = 0; i < data.Length; i++) data[i] = Encode(w.Buffer[i], precision);
        return new LowPrecisionMatrix(rows, cols, data, precision);
    }

    /// <summary>y = W·x, with W up-cast from 16-bit and accumulated in fp32.</summary>
    public DenseTensor<float> MatVec(DenseTensor<float> x)
    {
        if (x.Length != Cols) throw new ArgumentException($"x has {x.Length} elements; expected {Cols}.");
        var y = new DenseTensor<float>(Rows);
        for (int r = 0; r < Rows; r++)
        {
            float s = 0f;
            int baseIdx = r * Cols;
            for (int c = 0; c < Cols; c++) s += Decode(_data[baseIdx + c], _prec) * x.Buffer[c];
            y.Buffer[r] = s;
        }
        return y;
    }

    /// <summary>Reconstructs the fp32 matrix (lossy by the 16-bit rounding).</summary>
    public DenseTensor<float> ToFloat()
    {
        var w = new DenseTensor<float>(Rows, Cols);
        for (int i = 0; i < _data.Length; i++) w.Buffer[i] = Decode(_data[i], _prec);
        return w;
    }

    private static ushort Encode(float f, LowPrecision p) =>
        p == LowPrecision.Fp16 ? BitConverter.HalfToUInt16Bits((Half)f) : BFloat16.FromFloat(f).Bits;

    private static float Decode(ushort u, LowPrecision p) =>
        p == LowPrecision.Fp16 ? (float)BitConverter.UInt16BitsToHalf(u) : new BFloat16(u).ToFloat();
}
