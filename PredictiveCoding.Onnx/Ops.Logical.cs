namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// Ops.Logical — attention-mask kernels: Equal, Not, Where, Expand, Cast.
//
// ONNX bool tensors are represented here as int64 tensors holding 0/1 (the
// runtime has only two payload kinds — float Data and int64 I64), so the
// comparison ops emit int64 and Not/Where read the condition as int64. These
// ops run on SMALL mask/shape tensors (transformer attention-mask plumbing:
// Equal → Not/Cast → Expand → Where), so each one takes the simple
// output-index → per-input-offset broadcast walk (right-aligned numpy rules,
// stride 0 on stretched/missing dims) shared via BroadcastOffset below —
// clarity over speed. Shape/strides come from the BroadcastShape /
// BroadcastStrides helpers in Ops.Elementwise.cs, visible here because all
// kernels live in one partial class.
// ─────────────────────────────────────────────────────────────────────────────

public static partial class Ops
{
    /// <summary>Equal (opset 7+): numpy-broadcast element-wise equality. Both inputs
    /// must share a payload kind (both float or both int64); the output is a bool
    /// tensor, i.e. int64 holding 0/1. Float comparison is exact IEEE ==, so NaN
    /// never equals anything (including itself).</summary>
    public static Tensor Equal(Tensor a, Tensor b)
    {
        if (a.IsInt64 != b.IsInt64)
            throw new ArgumentException(
                $"Equal inputs must share a payload kind, got {PayloadKind(a)} {a.ShapeString} vs {PayloadKind(b)} {b.ShapeString}.");

        int[] outShape = BroadcastShape(a.Shape, b.Shape, "Equal");
        int[] sa = BroadcastStrides(a, outShape);
        int[] sb = BroadcastStrides(b, outShape);

        var dst = new long[Tensor.CheckedProduct(outShape)];
        if (a.IsInt64)
        {
            long[] xa = a.I64!, xb = b.I64!;
            for (int i = 0; i < dst.Length; i++)
                dst[i] = xa[BroadcastOffset(i, outShape, sa)] == xb[BroadcastOffset(i, outShape, sb)] ? 1L : 0L;
        }
        else
        {
            float[] xa = a.Data, xb = b.Data;
            for (int i = 0; i < dst.Length; i++)
                dst[i] = xa[BroadcastOffset(i, outShape, sa)] == xb[BroadcastOffset(i, outShape, sb)] ? 1L : 0L;
        }
        return new Tensor(outShape, dst);
    }

    /// <summary>Not (opset 1+): logical negation of a bool (int64 0/1) tensor —
    /// 1 where the value is 0, else 0. A float input is a graph error.</summary>
    public static Tensor Not(Tensor x)
    {
        long[] src = x.I64 ?? throw new ArgumentException(
            $"Not expects a bool (int64) tensor, got float {x.ShapeString}.");

        var dst = new long[src.Length];
        for (int i = 0; i < dst.Length; i++)
            dst[i] = src[i] == 0 ? 1L : 0L;
        return new Tensor(x.Shape, dst);
    }

    /// <summary>Where (opset 9+): element-wise select — cond != 0 picks from
    /// <paramref name="x"/>, else from <paramref name="y"/>. The condition must be a
    /// bool (int64 0/1) tensor; <paramref name="x"/> and <paramref name="y"/> must
    /// share a payload kind, which the output follows. All three shapes are
    /// numpy-broadcast together.</summary>
    public static Tensor Where(Tensor cond, Tensor x, Tensor y)
    {
        if (!cond.IsInt64)
            throw new ArgumentException(
                $"Where condition must be a bool (int64) tensor, got float {cond.ShapeString}.");
        if (x.IsInt64 != y.IsInt64)
            throw new ArgumentException(
                $"Where branches must share a payload kind, got {PayloadKind(x)} {x.ShapeString} vs {PayloadKind(y)} {y.ShapeString}.");

        int[] outShape = BroadcastShape(BroadcastShape(cond.Shape, x.Shape, "Where"), y.Shape, "Where");
        int[] sc = BroadcastStrides(cond, outShape);
        int[] sx = BroadcastStrides(x, outShape);
        int[] sy = BroadcastStrides(y, outShape);

        var result = NewLike(x, outShape);
        if (x.IsInt64) WhereCore(cond.I64!, x.I64!, y.I64!, result.I64!, outShape, sc, sx, sy);
        else WhereCore(cond.I64!, x.Data, y.Data, result.Data, outShape, sc, sx, sy);
        return result;
    }

    /// <summary>Expand (opset 8+): broadcast-copies <paramref name="x"/> to the
    /// BIDIRECTIONAL broadcast of its shape and the 1-D int64 <paramref name="shape"/>
    /// tensor (numpy rules: ranks right-aligned, a 1 on either side stretches to the
    /// other dim, anything else must match). Works for both payload kinds.</summary>
    public static Tensor Expand(Tensor x, Tensor shape)
    {
        long[] requested = RequireI64(shape, "Expand 'shape' input");
        var dims = new int[requested.Length];
        for (int i = 0; i < dims.Length; i++)
        {
            if (requested[i] < 0)
                throw new ArgumentException(
                    $"Expand shape dim {i} is {requested[i]}; negative dims are not allowed.");
            dims[i] = checked((int)requested[i]);
        }

        int[] outShape = BroadcastShape(x.Shape, dims, "Expand");
        int[] sx = BroadcastStrides(x, outShape);

        var result = NewLike(x, outShape);
        if (x.IsInt64) BroadcastCopy(x.I64!, result.I64!, outShape, sx);
        else BroadcastCopy(x.Data, result.Data, outShape, sx);
        return result;
    }

    /// <summary>Cast (opset 6+): converts to the ONNX TensorProto.DataType given by
    /// <paramref name="to"/>. Supported targets: FLOAT=1, INT32=6 (widened to this
    /// runtime's int64 payload), INT64=7, BOOL=9 (int64 0/1, value != 0 → 1; a float
    /// NaN is != 0 and casts to 1). Float → integer truncates toward zero via the
    /// plain C# (long) cast, NaN/Inf included. A cast that keeps the payload kind
    /// returns a new tensor sharing the same backing array.</summary>
    public static Tensor Cast(Tensor x, int to)
    {
        switch (to)
        {
            case 1:                                  // FLOAT
            {
                if (!x.IsInt64) return new Tensor(x.Shape, x.Data);
                long[] src = x.I64!;
                var dst = new float[src.Length];
                for (int i = 0; i < dst.Length; i++) dst[i] = src[i];
                return new Tensor(x.Shape, dst);
            }
            case 6:                                  // INT32 — widened to int64
            case 7:                                  // INT64
            {
                if (x.IsInt64) return new Tensor(x.Shape, x.I64!);
                float[] src = x.Data;
                var dst = new long[src.Length];
                for (int i = 0; i < dst.Length; i++) dst[i] = (long)src[i];
                return new Tensor(x.Shape, dst);
            }
            case 9:                                  // BOOL — int64 0/1
            {
                var dst = new long[x.Count];
                if (x.IsInt64)
                {
                    long[] src = x.I64!;
                    for (int i = 0; i < dst.Length; i++) dst[i] = src[i] != 0 ? 1L : 0L;
                }
                else
                {
                    float[] src = x.Data;
                    for (int i = 0; i < dst.Length; i++) dst[i] = src[i] != 0f ? 1L : 0L;
                }
                return new Tensor(x.Shape, dst);
            }
            default:
                throw new NotSupportedException(
                    $"Cast to ONNX data type {to} is not supported (supported: FLOAT=1, INT32=6, INT64=7, BOOL=9).");
        }
    }

    // ── the shared broadcast walk ────────────────────────────────────────────

    /// <summary>Maps a flat output index to an input element offset by decomposing it
    /// into per-axis coordinates over <paramref name="outShape"/> and dotting them with
    /// the input's broadcast strides (0 along stretched/missing dims, so those axes
    /// re-read the same input values).</summary>
    private static int BroadcastOffset(int flat, int[] outShape, int[] strides)
    {
        int off = 0;
        for (int ax = outShape.Length - 1; ax >= 0; ax--)
        {
            off += flat % outShape[ax] * strides[ax];
            flat /= outShape[ax];
        }
        return off;
    }

    /// <summary>The select loop behind <see cref="Where"/>, generic over the payload
    /// element type of the two branches.</summary>
    private static void WhereCore<T>(long[] cond, T[] x, T[] y, T[] dst,
                                     int[] outShape, int[] sc, int[] sx, int[] sy)
    {
        for (int i = 0; i < dst.Length; i++)
            dst[i] = cond[BroadcastOffset(i, outShape, sc)] != 0
                   ? x[BroadcastOffset(i, outShape, sx)]
                   : y[BroadcastOffset(i, outShape, sy)];
    }

    /// <summary>The copy loop behind <see cref="Expand"/>: every output element reads
    /// its broadcast source element, for either payload type.</summary>
    private static void BroadcastCopy<T>(T[] src, T[] dst, int[] outShape, int[] strides)
    {
        for (int i = 0; i < dst.Length; i++)
            dst[i] = src[BroadcastOffset(i, outShape, strides)];
    }

    /// <summary>"float" or "int64", for payload-kind mismatch messages.</summary>
    private static string PayloadKind(Tensor t) => t.IsInt64 ? "int64" : "float";
}
