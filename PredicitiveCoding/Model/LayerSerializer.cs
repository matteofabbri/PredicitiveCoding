using System.Buffers.Binary;
using System.Runtime.InteropServices;
using PredictiveCoding.Core;

namespace PredictiveCoding.Model;

// ─────────────────────────────────────────────────────────────────────────────
// LayerSerializer
//
// Flat binary layout for a Layer's mutable state:
//
//  Offset  Size   Field
//  ──────  ─────  ─────────────────────────────────────────────────────────────
//  0       4      Size       (int32 LE)
//  4       4      LowerSize  (int32 LE)
//  8       4      Precision  (float32 LE)
//  12      4      WeightDecay
//  16      4      StateLearningRate
//  20      4      WeightLearningRate
//  24      1      IsClamped  (0/1)
//  25      1      flags      bit0=hasDiagPrec  bit1=hasPriorMean
//                            bit2=halfWeights  bit3=bf16Weights  bit4=int8Weights
//  26      2      (padding, align to 4)
//  28      Size*4             Mu
//  +       Size*4             PredictionError
//  if LowerSize > 0, the weights in the persisted precision (roadmap Ph. 6-7):
//    float32 (default):  LowerSize*Size*4   GenerativeWeights (row-major)
//    bit2 Half:          LowerSize*Size*2   — on load the Half copy is
//                        populated too, so mixed inference skips re-quantizing
//    bit3 bfloat16:      LowerSize*Size*2   — fp32 range, 8-bit mantissa
//    bit4 int8 affine:   LowerSize*4 scales + LowerSize zero-points +
//                        LowerSize*Size codes   (per-row scale + zero-point)
//  then, always float32:
//    +     LowerSize*4        Bias
//    +     LowerSize*4        PreActivation
//    +     LowerSize*4        Prediction
//  if hasDiagPrec:
//    +     Size*4             DiagonalPrecision
//  if hasPriorMean:
//    +     Size*4             PriorMean
// ─────────────────────────────────────────────────────────────────────────────

internal static class LayerSerializer
{
    private const int FixedHeaderBytes = 4 + 4 + 4 + 4 + 4 + 4 + 1 + 1 + 2;

    // ── Allocate + Deserialize ────────────────────────────────────────────────

    /// <summary>
    /// Peeks the binary header, allocates a <see cref="Layer"/> with the correct
    /// tensor shapes, then deserializes all fields into it.
    /// </summary>
    public static Layer AllocateAndDeserialize(ReadOnlySpan<byte> span, int index, ActivationType activation)
    {
        // Header peek (no advancing): offsets 0,4,25 are fixed by the layout.
        int  size      = BinaryPrimitives.ReadInt32LittleEndian(span);
        int  lowerSize = BinaryPrimitives.ReadInt32LittleEndian(span.Slice(4));
        byte flags     = span[25];
        bool hasDiag   = (flags & 1) != 0;
        bool hasPrior  = (flags & 2) != 0;

        var layer = new Layer
        {
            Index          = index,
            Size           = size,
            LowerSize      = lowerSize,
            ActivationType = activation,
        };
        layer.AllocateTensors(hasDiag, hasPrior);
        Deserialize(span, layer);
        return layer;
    }

    // ── Serialize ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Serializes the layer. <paramref name="precision"/> selects how the
    /// generative weights are stored (roadmap Phases 6-7) — rounded/quantized
    /// from the float32 masters; everything else always stays float32.
    /// </summary>
    public static byte[] Serialize(Layer layer, WeightStoragePrecision precision = WeightStoragePrecision.Float32)
    {
        bool hasDiag  = layer.DiagonalPrecision != null;
        bool hasPrior = layer.PriorMean         != null;
        if (layer.LowerSize == 0) precision = WeightStoragePrecision.Float32;

        int totalBytes = ComputeByteLength(layer.Size, layer.LowerSize, hasDiag, hasPrior, precision);
        var buf = new byte[totalBytes];
        var span = buf.AsSpan();
        int pos = 0;

        byte precisionFlags = precision switch
        {
            WeightStoragePrecision.Half     => 4,
            WeightStoragePrecision.BFloat16 => 8,
            WeightStoragePrecision.Int8     => 16,
            _                               => 0,
        };

        WriteInt  (span, ref pos, layer.Size);
        WriteInt  (span, ref pos, layer.LowerSize);
        WriteFloat(span, ref pos, layer.Precision);
        WriteFloat(span, ref pos, layer.WeightDecay);
        WriteFloat(span, ref pos, layer.StateLearningRate);
        WriteFloat(span, ref pos, layer.WeightLearningRate);
        span[pos++] = layer.IsClamped ? (byte)1 : (byte)0;
        span[pos++] = (byte)((hasDiag ? 1 : 0) | (hasPrior ? 2 : 0) | precisionFlags);
        pos += 2; // padding

        WriteFloats(span, ref pos, layer.Mu.Buffer.AsSpan());
        WriteFloats(span, ref pos, layer.PredictionError.Buffer.AsSpan());

        if (layer.LowerSize > 0)
        {
            switch (precision)
            {
                case WeightStoragePrecision.Half:
                    WriteHalves(span, ref pos, TensorUtils.ToHalf(layer.GenerativeWeights!).Buffer.AsSpan());
                    break;
                case WeightStoragePrecision.BFloat16:
                {
                    var bits = new ushort[layer.LowerSize * layer.Size];
                    TensorUtils.FloatToBf16Bits(layer.GenerativeWeights!.Buffer.AsSpan(), bits);
                    WriteUShorts(span, ref pos, bits);
                    break;
                }
                case WeightStoragePrecision.Int8:
                    WriteInt8Weights(span, ref pos, layer.GenerativeWeights!.Buffer, layer.LowerSize, layer.Size);
                    break;
                default:
                    WriteFloats(span, ref pos, layer.GenerativeWeights!.Buffer.AsSpan());
                    break;
            }
            WriteFloats(span, ref pos, layer.Bias!.Buffer.AsSpan());
            WriteFloats(span, ref pos, layer.PreActivation!.Buffer.AsSpan());
            WriteFloats(span, ref pos, layer.Prediction!.Buffer.AsSpan());
        }

        if (hasDiag)  WriteFloats(span, ref pos, layer.DiagonalPrecision!.Buffer.AsSpan());
        if (hasPrior) WriteFloats(span, ref pos, layer.PriorMean!.Buffer.AsSpan());

        return buf;
    }

    // ── Deserialize ───────────────────────────────────────────────────────────

    public static void Deserialize(ReadOnlySpan<byte> span, Layer target)
    {
        int pos = 0;

        int size      = ReadInt  (span, ref pos);
        int lowerSize = ReadInt  (span, ref pos);

        if (size != target.Size || lowerSize != target.LowerSize)
            throw new InvalidDataException(
                $"Layer size mismatch: stored ({size},{lowerSize}) " +
                $"vs target ({target.Size},{target.LowerSize}).");

        target.Precision          = ReadFloat(span, ref pos);
        target.WeightDecay        = ReadFloat(span, ref pos);
        target.StateLearningRate  = ReadFloat(span, ref pos);
        target.WeightLearningRate = ReadFloat(span, ref pos);
        target.IsClamped          = span[pos++] != 0;
        byte flags                = span[pos++];
        pos += 2; // padding

        bool hasDiag  = (flags & 1) != 0;
        bool hasPrior = (flags & 2) != 0;
        bool half     = (flags & 4) != 0;
        bool bf16     = (flags & 8) != 0;
        bool int8     = (flags & 16) != 0;

        ReadFloats(span, ref pos, target.Mu.Buffer.AsSpan());
        ReadFloats(span, ref pos, target.PredictionError.Buffer.AsSpan());

        if (lowerSize > 0)
        {
            if (half)
            {
                // Half-persisted record (roadmap Phase 6): the stored Half
                // weights become BOTH the mixed-ready copy and — widened —
                // the float masters, so behaviour matches a live-quantized net.
                var halfWeights = TensorUtils.ToHalf(target.GenerativeWeights!);   // right shape; values overwritten next
                ReadHalves(span, ref pos, halfWeights.Buffer.AsSpan());
                target.GenerativeWeightsHalf = halfWeights;
                TensorUtils.ToFloat(halfWeights).Buffer.AsSpan().CopyTo(target.GenerativeWeights!.Buffer.AsSpan());
            }
            else if (bf16)
            {
                // bf16-persisted record (roadmap Phase 7): widened straight
                // into the masters; the mixed Half copy stays on-demand.
                var bits = new ushort[lowerSize * size];
                ReadUShorts(span, ref pos, bits);
                TensorUtils.Bf16BitsToFloat(bits, target.GenerativeWeights!.Buffer.AsSpan());
            }
            else if (int8)
            {
                // int8 affine per-row (roadmap Phase 7): dequantized into the
                // masters — w̃ = (q − z) · s per row.
                ReadInt8Weights(span, ref pos, target.GenerativeWeights!.Buffer, lowerSize, size);
            }
            else
            {
                ReadFloats(span, ref pos, target.GenerativeWeights!.Buffer.AsSpan());
            }
            ReadFloats(span, ref pos, target.Bias!.Buffer.AsSpan());
            ReadFloats(span, ref pos, target.PreActivation!.Buffer.AsSpan());
            ReadFloats(span, ref pos, target.Prediction!.Buffer.AsSpan());
        }

        if (hasDiag)  ReadFloats(span, ref pos, target.DiagonalPrecision!.Buffer.AsSpan());
        if (hasPrior) ReadFloats(span, ref pos, target.PriorMean!.Buffer.AsSpan());
    }

    // ── Byte-length calculation ───────────────────────────────────────────────

    public static int ComputeByteLength(int size, int lowerSize, bool hasDiag, bool hasPrior,
                                        WeightStoragePrecision precision = WeightStoragePrecision.Float32)
    {
        int n = FixedHeaderBytes;
        n += size * 4 * 2; // Mu + PredictionError
        if (lowerSize > 0)
        {
            n += precision switch
            {
                WeightStoragePrecision.Half     => lowerSize * size * 2,
                WeightStoragePrecision.BFloat16 => lowerSize * size * 2,
                WeightStoragePrecision.Int8     => lowerSize * 4 + lowerSize + lowerSize * size,   // scales + zero-points + codes
                _                               => lowerSize * size * 4,
            };
            n += lowerSize * 3 * 4; // b + preAct + pred
        }
        if (hasDiag)  n += size * 4;
        if (hasPrior) n += size * 4;
        return n;
    }

    // ── int8 affine per-row weights (roadmap Phase 7) ─────────────────────────
    //
    // Layout: float scales[rows], sbyte zeroPoints[rows], sbyte codes[rows·cols].
    // Quantization:  s = (max − min) / 255,  z = round(−min/s) − 128,
    //                q = clamp(round(w/s) + z, −128, 127);   dequant  w̃ = (q − z)·s.
    // Weight PTQ needs no activation calibration; the affine zero-point handles
    // asymmetric rows (for near-symmetric weights it lands near 0 by itself).

    private static void WriteInt8Weights(Span<byte> span, ref int pos, float[] w, int rows, int cols)
    {
        var scales = new float[rows];
        var zeros  = new sbyte[rows];
        var codes  = new sbyte[rows * cols];

        for (int r = 0; r < rows; r++)
        {
            float min = float.MaxValue, max = float.MinValue;
            for (int j = 0; j < cols; j++)
            {
                float v = w[r * cols + j];
                if (v < min) min = v;
                if (v > max) max = v;
            }

            float s;
            int   z;
            if (max - min < 1e-12f)
            {
                // Constant row: symmetric scale around the single value.
                s = MathF.Max(MathF.Abs(min), 1e-12f) / 127f;
                z = 0;
            }
            else
            {
                s = (max - min) / 255f;
                z = Math.Clamp((int)MathF.Round(-min / s) - 128, -128, 127);
            }

            scales[r] = s;
            zeros[r]  = (sbyte)z;
            for (int j = 0; j < cols; j++)
                codes[r * cols + j] = (sbyte)Math.Clamp((int)MathF.Round(w[r * cols + j] / s) + z, -128, 127);
        }

        WriteFloats(span, ref pos, scales);
        WriteSBytes(span, ref pos, zeros);
        WriteSBytes(span, ref pos, codes);
    }

    private static void ReadInt8Weights(ReadOnlySpan<byte> span, ref int pos, float[] w, int rows, int cols)
    {
        var scales = new float[rows];
        var zeros  = new sbyte[rows];
        var codes  = new sbyte[rows * cols];
        ReadFloats(span, ref pos, scales);
        ReadSBytes(span, ref pos, zeros);
        ReadSBytes(span, ref pos, codes);

        for (int r = 0; r < rows; r++)
        {
            float s = scales[r];
            int   z = zeros[r];
            for (int j = 0; j < cols; j++)
                w[r * cols + j] = (codes[r * cols + j] - z) * s;
        }
    }

    // ── Primitive writers ─────────────────────────────────────────────────────

    private static void WriteInt(Span<byte> span, ref int pos, int value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(span.Slice(pos), value);
        pos += 4;
    }

    private static void WriteFloat(Span<byte> span, ref int pos, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(span.Slice(pos), BitConverter.SingleToInt32Bits(value));
        pos += 4;
    }

    private static void WriteFloats(Span<byte> span, ref int pos, Span<float> floats)
    {
        var bytes = MemoryMarshal.Cast<float, byte>(floats);
        bytes.CopyTo(span.Slice(pos));
        pos += bytes.Length;
    }

    private static void WriteHalves(Span<byte> span, ref int pos, Span<Half> halves)
    {
        var bytes = MemoryMarshal.Cast<Half, byte>(halves);
        bytes.CopyTo(span.Slice(pos));
        pos += bytes.Length;
    }

    private static void WriteUShorts(Span<byte> span, ref int pos, Span<ushort> values)
    {
        var bytes = MemoryMarshal.Cast<ushort, byte>(values);
        bytes.CopyTo(span.Slice(pos));
        pos += bytes.Length;
    }

    private static void WriteSBytes(Span<byte> span, ref int pos, Span<sbyte> values)
    {
        var bytes = MemoryMarshal.Cast<sbyte, byte>(values);
        bytes.CopyTo(span.Slice(pos));
        pos += bytes.Length;
    }

    // ── Primitive readers ─────────────────────────────────────────────────────

    private static int ReadInt(ReadOnlySpan<byte> span, ref int pos)
    {
        int v = BinaryPrimitives.ReadInt32LittleEndian(span.Slice(pos));
        pos += 4;
        return v;
    }

    private static float ReadFloat(ReadOnlySpan<byte> span, ref int pos)
    {
        float v = BitConverter.Int32BitsToSingle(
            BinaryPrimitives.ReadInt32LittleEndian(span.Slice(pos)));
        pos += 4;
        return v;
    }

    private static void ReadFloats(ReadOnlySpan<byte> span, ref int pos, Span<float> target)
    {
        var dst = MemoryMarshal.Cast<float, byte>(target);
        span.Slice(pos, dst.Length).CopyTo(dst);
        pos += dst.Length;
    }

    private static void ReadHalves(ReadOnlySpan<byte> span, ref int pos, Span<Half> target)
    {
        var dst = MemoryMarshal.Cast<Half, byte>(target);
        span.Slice(pos, dst.Length).CopyTo(dst);
        pos += dst.Length;
    }

    private static void ReadUShorts(ReadOnlySpan<byte> span, ref int pos, Span<ushort> target)
    {
        var dst = MemoryMarshal.Cast<ushort, byte>(target);
        span.Slice(pos, dst.Length).CopyTo(dst);
        pos += dst.Length;
    }

    private static void ReadSBytes(ReadOnlySpan<byte> span, ref int pos, Span<sbyte> target)
    {
        var dst = MemoryMarshal.Cast<sbyte, byte>(target);
        span.Slice(pos, dst.Length).CopyTo(dst);
        pos += dst.Length;
    }
}
