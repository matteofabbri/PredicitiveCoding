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
//  25      1      flags      bit0=hasDiagPrec  bit1=hasPriorMean  bit2=halfWeights
//  26      2      (padding, align to 4)
//  28      Size*4             Mu
//  +       Size*4             PredictionError
//  if LowerSize > 0:
//    +     LowerSize*Size*4   GenerativeWeights  (row-major, float32)
//          — or, when bit2 (halfWeights) is set —
//    +     LowerSize*Size*2   GenerativeWeights  (row-major, Half — roadmap
//          Phase 6: the record stores the weights at half size, and on load the
//          Half copy is populated too, so mixed inference skips re-quantizing)
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
    /// Serializes the layer. With <paramref name="halfWeights"/> the generative
    /// weights are stored as Half (2 bytes each — roadmap Phase 6), rounded
    /// from the float32 masters; everything else stays float32.
    /// </summary>
    public static byte[] Serialize(Layer layer, bool halfWeights = false)
    {
        bool hasDiag  = layer.DiagonalPrecision != null;
        bool hasPrior = layer.PriorMean         != null;
        bool half     = halfWeights && layer.LowerSize > 0;

        int totalBytes = ComputeByteLength(layer.Size, layer.LowerSize, hasDiag, hasPrior, half);
        var buf = new byte[totalBytes];
        var span = buf.AsSpan();
        int pos = 0;

        WriteInt  (span, ref pos, layer.Size);
        WriteInt  (span, ref pos, layer.LowerSize);
        WriteFloat(span, ref pos, layer.Precision);
        WriteFloat(span, ref pos, layer.WeightDecay);
        WriteFloat(span, ref pos, layer.StateLearningRate);
        WriteFloat(span, ref pos, layer.WeightLearningRate);
        span[pos++] = layer.IsClamped ? (byte)1 : (byte)0;
        span[pos++] = (byte)((hasDiag ? 1 : 0) | (hasPrior ? 2 : 0) | (half ? 4 : 0));
        pos += 2; // padding

        WriteFloats(span, ref pos, layer.Mu.Buffer.AsSpan());
        WriteFloats(span, ref pos, layer.PredictionError.Buffer.AsSpan());

        if (layer.LowerSize > 0)
        {
            if (half)
                WriteHalves(span, ref pos, TensorUtils.ToHalf(layer.GenerativeWeights!).Buffer.AsSpan());
            else
                WriteFloats(span, ref pos, layer.GenerativeWeights!.Buffer.AsSpan());
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

    public static int ComputeByteLength(int size, int lowerSize, bool hasDiag, bool hasPrior, bool halfWeights = false)
    {
        int n = FixedHeaderBytes;
        n += size * 4 * 2; // Mu + PredictionError
        if (lowerSize > 0)
            n += lowerSize * size * (halfWeights ? 2 : 4) + lowerSize * 3 * 4; // W + b + preAct + pred
        if (hasDiag)  n += size * 4;
        if (hasPrior) n += size * 4;
        return n;
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
}
