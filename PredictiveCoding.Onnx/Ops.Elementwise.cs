namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// Ops.Elementwise — pointwise math kernels: Add, Sub, Mul, Div, Sigmoid, Relu,
// Exp, Sqrt, PRelu, Softmax.
//
// Every binary op flows through ONE broadcast engine (Binary<TOp>): shapes are
// right-aligned numpy-style, 1-dims stretch, and each output element reads its
// operands through per-axis strides that are 0 along broadcast axes.
// Identically-shaped operands — the overwhelmingly common case (residual adds,
// sigmoid gates) — skip the stride machinery and run one flat array loop.
// Per-op code lives in tiny struct functors: the JIT clones the generic loops
// per functor and inlines Apply, so each op gets a dedicated tight loop without
// duplicating the machinery. Big tensors (YOLOX 640×640 feature maps are
// multi-million-element) split into parallel blocks on the thread pool.
// ─────────────────────────────────────────────────────────────────────────────

public static partial class Ops
{
    /// <summary>Add (opset 7+): elementwise a + b with numpy-style broadcasting.</summary>
    public static Tensor Add(Tensor a, Tensor b) => Binary<AddF>(a, b, "Add");

    /// <summary>Sub (opset 7+): elementwise a − b with numpy-style broadcasting.</summary>
    public static Tensor Sub(Tensor a, Tensor b) => Binary<SubF>(a, b, "Sub");

    /// <summary>Mul (opset 7+): elementwise a · b with numpy-style broadcasting.</summary>
    public static Tensor Mul(Tensor a, Tensor b) => Binary<MulF>(a, b, "Mul");

    /// <summary>Div (opset 7+): elementwise a / b with numpy-style broadcasting
    /// (IEEE semantics: x/0 gives ±infinity, 0/0 gives NaN).</summary>
    public static Tensor Div(Tensor a, Tensor b) => Binary<DivF>(a, b, "Div");

    /// <summary>Sigmoid: 1 / (1 + e^−v) per element.</summary>
    public static Tensor Sigmoid(Tensor x) => Map<SigmoidF>(x, "Sigmoid");

    /// <summary>Relu: max(v, 0) per element.</summary>
    public static Tensor Relu(Tensor x) => Map<ReluF>(x, "Relu");

    /// <summary>Exp: e^v per element.</summary>
    public static Tensor Exp(Tensor x) => Map<ExpF>(x, "Exp");

    /// <summary>Sqrt: √v per element (NaN for negative input, per IEEE).</summary>
    public static Tensor Sqrt(Tensor x) => Map<SqrtF>(x, "Sqrt");

    /// <summary>PRelu: v &lt; 0 ? slope·v : v per element. <paramref name="slope"/>
    /// broadcasts UNIDIRECTIONALLY into <paramref name="x"/> (typically per-channel
    /// [C] or [C,1,1] against an NCHW map), so the output always has x's shape.</summary>
    public static Tensor PRelu(Tensor x, Tensor slope)
    {
        RequireFloat(x, "PRelu");
        RequireFloat(slope, "PRelu");

        // ONNX PRelu broadcasting is unidirectional — slope stretches into x, never
        // the other way — so the mutual broadcast shape must be exactly x's shape.
        if (!x.Shape.AsSpan().SequenceEqual(slope.Shape))
        {
            int[] outShape = BroadcastShape(x.Shape, slope.Shape, "PRelu");
            if (!outShape.AsSpan().SequenceEqual(x.Shape))
                throw new ArgumentException(
                    $"PRelu slope {slope.ShapeString} must broadcast into x {x.ShapeString}; their mutual broadcast [{string.Join(",", outShape)}] would change x's shape.");
        }
        return Binary<PReluF>(x, slope, "PRelu");
    }

    /// <summary>Softmax (opset 13 semantics): e^(v−max) / Σ e^(v−max) along the SINGLE
    /// axis <paramref name="axis"/> (negative counts from the end), computed
    /// independently for every combination of the other indices. Subtracting the
    /// per-slice max keeps large logits from overflowing e^v.</summary>
    public static Tensor Softmax(Tensor x, int axis)
    {
        RequireFloat(x, "Softmax");
        int rank = x.Rank;
        if (axis < -rank || axis >= rank)
            throw new ArgumentException($"Softmax axis {axis} out of range for rank {rank} (allowed [{-rank},{rank - 1}]).");
        if (axis < 0) axis += rank;

        int axisLen = x.Shape[axis];
        int inner = 1;                               // elements per step along the axis
        for (int d = axis + 1; d < rank; d++) inner *= x.Shape[d];
        int outer = 1;                               // independent blocks before the axis
        for (int d = 0; d < axis; d++) outer *= x.Shape[d];

        var y = new Tensor(x.Shape);
        float[] src = x.Data, dst = y.Data;
        if (src.Length == 0) return y;

        void DoOuter(int o)
        {
            int baseOff = o * axisLen * inner;
            for (int i = 0; i < inner; i++)
            {
                int off = baseOff + i;               // slice runs off, off+inner, off+2·inner, …

                float max = float.NegativeInfinity;
                for (int k = 0, p = off; k < axisLen; k++, p += inner)
                    if (src[p] > max) max = src[p];

                float sum = 0f;
                for (int k = 0, p = off; k < axisLen; k++, p += inner)
                {
                    float e = MathF.Exp(src[p] - max);
                    dst[p] = e;
                    sum += e;
                }

                float invSum = 1f / sum;
                for (int k = 0, p = off; k < axisLen; k++, p += inner)
                    dst[p] *= invSum;
            }
        }

        if (src.Length > ParallelCutoff && outer > 1) Parallel.For(0, outer, DoOuter);
        else for (int o = 0; o < outer; o++) DoOuter(o);
        return y;
    }

    // ── the shared broadcast engine ──────────────────────────────────────────

    /// <summary>Element count above which pointwise loops split into parallel blocks.</summary>
    private const int ParallelCutoff = 1 << 18;      // ≈262k elements

    /// <summary>Elements per parallel block — big enough that the loop body dwarfs the
    /// thread-pool dispatch, small enough to load-balance across cores.</summary>
    private const int ParallelBlock = 1 << 16;

    /// <summary>Guards the float-only math kernels: int64 (shape/index) tensors must
    /// fail loudly instead of reading the empty <see cref="Tensor.Data"/> array.</summary>
    private static void RequireFloat(Tensor t, string op)
    {
        if (t.IsInt64)
            throw new NotSupportedException($"{op} supports float32 tensors only, got an int64 tensor {t.ShapeString}.");
    }

    /// <summary>Numpy-style broadcast of two shapes: right-aligned, each dim pair must
    /// match or contain a 1 (which stretches to the other). Also reused by MatMul for
    /// its leading batch dims.</summary>
    internal static int[] BroadcastShape(int[] a, int[] b, string op)
    {
        int rank = Math.Max(a.Length, b.Length);
        var result = new int[rank];
        for (int i = 0; i < rank; i++)
        {
            int ia = i - (rank - a.Length);          // right-aligned source indices
            int ib = i - (rank - b.Length);
            int da = ia < 0 ? 1 : a[ia];
            int db = ib < 0 ? 1 : b[ib];
            if (da == db) result[i] = da;
            else if (da == 1) result[i] = db;
            else if (db == 1) result[i] = da;
            else throw new ArgumentException(
                $"{op}: shapes [{string.Join(",", a)}] and [{string.Join(",", b)}] are not broadcast-compatible (dim {da} vs {db}).");
        }
        return result;
    }

    /// <summary>Per-output-axis element strides for reading <paramref name="t"/> at an
    /// output coordinate: 0 along broadcast axes (stretched 1-dims and missing leading
    /// dims), the tensor's own row-major stride elsewhere.</summary>
    private static int[] BroadcastStrides(Tensor t, int[] outShape)
    {
        var strides = new int[outShape.Length];
        var own = t.Strides();
        int shift = outShape.Length - t.Rank;
        for (int i = 0; i < t.Rank; i++)
            if (t.Shape[i] != 1)
                strides[shift + i] = own[i];
        return strides;
    }

    /// <summary>The one binary engine behind Add/Sub/Mul/Div/PRelu: a flat loop when
    /// the shapes are identical, a strided broadcast walk otherwise; both go parallel
    /// once the tensor is big enough.</summary>
    private static Tensor Binary<TOp>(Tensor a, Tensor b, string op) where TOp : struct, IBinaryOp
    {
        RequireFloat(a, op);
        RequireFloat(b, op);
        TOp f = default;
        float[] xa = a.Data, xb = b.Data;

        // Fast path: identical shapes → one flat loop, no index arithmetic at all.
        if (a.Shape.AsSpan().SequenceEqual(b.Shape))
        {
            var y = new Tensor(a.Shape);
            float[] yd = y.Data;
            int n = yd.Length;
            if (n > ParallelCutoff)
            {
                Parallel.For(0, (n + ParallelBlock - 1) / ParallelBlock, blk =>
                {
                    int start = blk * ParallelBlock;
                    int end = n - start > ParallelBlock ? start + ParallelBlock : n;
                    for (int i = start; i < end; i++)
                        yd[i] = f.Apply(xa[i], xb[i]);
                });
            }
            else
            {
                for (int i = 0; i < n; i++)
                    yd[i] = f.Apply(xa[i], xb[i]);
            }
            return y;
        }

        // Broadcast path: walk the output row by row (a row = one run along the last
        // axis); each row's input offsets come from per-axis strides that are 0 along
        // broadcast axes, so stretched dims re-read the same operand values.
        int[] outShape = BroadcastShape(a.Shape, b.Shape, op);
        var result = new Tensor(outShape);
        float[] rd = result.Data;
        if (rd.Length == 0) return result;

        int rank = outShape.Length;
        int[] sa = BroadcastStrides(a, outShape);
        int[] sb = BroadcastStrides(b, outShape);
        int inner = outShape[rank - 1];
        int stepA = sa[rank - 1], stepB = sb[rank - 1];
        int rows = rd.Length / inner;

        void DoRow(int r)
        {
            int offA = 0, offB = 0;                  // decompose the row index into outer coords
            for (int ax = rank - 2, rem = r; ax >= 0; ax--)
            {
                int idx = rem % outShape[ax];
                rem /= outShape[ax];
                offA += idx * sa[ax];
                offB += idx * sb[ax];
            }
            int pos = r * inner;
            for (int c = 0; c < inner; c++, offA += stepA, offB += stepB)
                rd[pos + c] = f.Apply(xa[offA], xb[offB]);
        }

        if (rd.Length > ParallelCutoff && rows > 1) Parallel.For(0, rows, DoRow);
        else for (int r = 0; r < rows; r++) DoRow(r);
        return result;
    }

    /// <summary>The unary engine behind Sigmoid/Relu/Exp/Sqrt: one flat loop, split
    /// into parallel blocks for big tensors.</summary>
    private static Tensor Map<TOp>(Tensor x, string op) where TOp : struct, IUnaryOp
    {
        RequireFloat(x, op);
        var y = new Tensor(x.Shape);
        float[] src = x.Data, dst = y.Data;
        TOp f = default;
        int n = src.Length;
        if (n > ParallelCutoff)
        {
            Parallel.For(0, (n + ParallelBlock - 1) / ParallelBlock, blk =>
            {
                int start = blk * ParallelBlock;
                int end = n - start > ParallelBlock ? start + ParallelBlock : n;
                for (int i = start; i < end; i++)
                    dst[i] = f.Apply(src[i]);
            });
        }
        else
        {
            for (int i = 0; i < n; i++)
                dst[i] = f.Apply(src[i]);
        }
        return y;
    }

    // ── struct functors (JIT-specialised per op, Apply inlined) ──────────────

    private interface IUnaryOp { float Apply(float v); }
    private interface IBinaryOp { float Apply(float a, float b); }

    private readonly struct SigmoidF : IUnaryOp { public float Apply(float v) => 1f / (1f + MathF.Exp(-v)); }
    private readonly struct ReluF : IUnaryOp { public float Apply(float v) => MathF.Max(v, 0f); }
    private readonly struct ExpF : IUnaryOp { public float Apply(float v) => MathF.Exp(v); }
    private readonly struct SqrtF : IUnaryOp { public float Apply(float v) => MathF.Sqrt(v); }

    private readonly struct AddF : IBinaryOp { public float Apply(float a, float b) => a + b; }
    private readonly struct SubF : IBinaryOp { public float Apply(float a, float b) => a - b; }
    private readonly struct MulF : IBinaryOp { public float Apply(float a, float b) => a * b; }
    private readonly struct DivF : IBinaryOp { public float Apply(float a, float b) => a / b; }
    private readonly struct PReluF : IBinaryOp { public float Apply(float v, float s) => v < 0f ? s * v : v; }
}
