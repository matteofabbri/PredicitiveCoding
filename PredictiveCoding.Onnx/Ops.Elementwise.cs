namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// Ops.Elementwise — pointwise math kernels: Add, Sub, Mul, Div, Pow, Sigmoid,
// Relu, Exp, Sqrt, Erf, Clip, PRelu, Softmax.
//
// Every binary op flows through ONE broadcast engine (BinaryCore<T,TOp>): shapes
// are right-aligned numpy-style, 1-dims stretch, and each output element reads
// its operands through per-axis strides that are 0 along broadcast axes.
// Identically-shaped operands — the overwhelmingly common case (residual adds,
// sigmoid gates) — skip the stride machinery and run one flat array loop.
// Per-op code lives in tiny struct functors: the JIT clones the generic loops
// per functor and inlines Apply, so each op gets a dedicated tight loop without
// duplicating the machinery. Big tensors (YOLOX 640×640 feature maps are
// multi-million-element) split into parallel blocks on the thread pool.
// The engine is generic over the element type: Add/Sub/Mul/Div also accept
// int64 pairs (transformer attention-mask and position-id arithmetic) through
// the very same broadcast machinery, while mixing float and int64 operands
// throws — ONNX only ever mixes element types through an explicit Cast.
// ─────────────────────────────────────────────────────────────────────────────

public static partial class Ops
{
    /// <summary>Add (opset 7+): elementwise a + b with numpy-style broadcasting.
    /// Operands are both float32 or both int64 (mask/position arithmetic); mixing
    /// kinds throws.</summary>
    public static Tensor Add(Tensor a, Tensor b) => BinaryNumeric<AddF, AddL>(a, b, "Add");

    /// <summary>Sub (opset 7+): elementwise a − b with numpy-style broadcasting.
    /// Operands are both float32 or both int64; mixing kinds throws.</summary>
    public static Tensor Sub(Tensor a, Tensor b) => BinaryNumeric<SubF, SubL>(a, b, "Sub");

    /// <summary>Mul (opset 7+): elementwise a · b with numpy-style broadcasting.
    /// Operands are both float32 or both int64; mixing kinds throws.</summary>
    public static Tensor Mul(Tensor a, Tensor b) => BinaryNumeric<MulF, MulL>(a, b, "Mul");

    /// <summary>Div (opset 7+): elementwise a / b with numpy-style broadcasting.
    /// Float32 pairs use IEEE semantics (x/0 gives ±infinity, 0/0 gives NaN); int64
    /// pairs use C# integer division, truncating toward zero per the ONNX spec.</summary>
    public static Tensor Div(Tensor a, Tensor b) => BinaryNumeric<DivF, DivL>(a, b, "Div");

    /// <summary>Pow (opset 7+): elementwise a^b with numpy-style broadcasting, float32
    /// only. A single-value exponent equal to 2 — the LayerNorm variance square — takes
    /// a dedicated squaring path instead of the general MathF.Pow.</summary>
    public static Tensor Pow(Tensor a, Tensor b)
    {
        RequireFloat(a, "Pow");
        RequireFloat(b, "Pow");
        if (b.Count == 1 && b.Rank <= a.Rank && b.Data[0] == 2f)
            return Map<SquareF>(a, "Pow");           // b's 1-dims all stretch: output shape is a's
        return Binary<PowF>(a, b, "Pow");
    }

    /// <summary>Sigmoid: 1 / (1 + e^−v) per element.</summary>
    public static Tensor Sigmoid(Tensor x) => Map<SigmoidF>(x, "Sigmoid");

    /// <summary>Relu: max(v, 0) per element.</summary>
    public static Tensor Relu(Tensor x) => Map<ReluF>(x, "Relu");

    /// <summary>Exp: e^v per element.</summary>
    public static Tensor Exp(Tensor x) => Map<ExpF>(x, "Exp");

    /// <summary>Sqrt: √v per element (NaN for negative input, per IEEE).</summary>
    public static Tensor Sqrt(Tensor x) => Map<SqrtF>(x, "Sqrt");

    /// <summary>Erf: the Gauss error function per element (the exact-GELU building
    /// block), float32 only. Evaluated in double precision with absolute error
    /// below 1e-9 over all reals; odd symmetry is exact (erf(−v) == −erf(v)) and
    /// |v| ≥ 6 saturates to ±1 exactly. See <see cref="ErfDouble"/>.</summary>
    public static Tensor Erf(Tensor x) => Map<ErfF>(x, "Erf");

    /// <summary>Clip: clamps every element to [min, max], float32 only. Bounds come
    /// from the optional single-element tensors (opset 11+ input form; anything but
    /// exactly one element throws) and fall back to the attribute values when a tensor
    /// slot is null (opset ≤ 6 form — the dispatcher passes ±infinity when absent).
    /// Min is applied before max, so when min &gt; max the max bound wins, matching
    /// the ONNX spec.</summary>
    public static Tensor Clip(Tensor x, Tensor? min, Tensor? max, float attrMin, float attrMax)
    {
        RequireFloat(x, "Clip");

        float lo = attrMin;
        if (min is not null)
        {
            RequireFloat(min, "Clip");
            if (min.Count != 1)
                throw new ArgumentException($"Clip 'min' must hold exactly one element, got {min.ShapeString}.");
            lo = min.Data[0];
        }

        float hi = attrMax;
        if (max is not null)
        {
            RequireFloat(max, "Clip");
            if (max.Count != 1)
                throw new ArgumentException($"Clip 'max' must hold exactly one element, got {max.ShapeString}.");
            hi = max.Data[0];
        }

        return Map(x, "Clip", new ClipF(lo, hi));
    }

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

    /// <summary>Float-only front of the binary engine, for the ops (PRelu, Pow) whose
    /// semantics only exist for float tensors.</summary>
    private static Tensor Binary<TOp>(Tensor a, Tensor b, string op) where TOp : struct, IBinaryOp<float>
    {
        RequireFloat(a, op);
        RequireFloat(b, op);
        return BinaryCore<float, TOp>(a, b, a.Data, b.Data, op);
    }

    /// <summary>Kind-dispatching front of the binary engine for Add/Sub/Mul/Div:
    /// float32 pairs run the <typeparamref name="TF"/> functor, int64 pairs the
    /// <typeparamref name="TL"/> one — same broadcast machinery either way. Mixed
    /// kinds throw: ONNX only mixes element types through an explicit Cast.</summary>
    private static Tensor BinaryNumeric<TF, TL>(Tensor a, Tensor b, string op)
        where TF : struct, IBinaryOp<float>
        where TL : struct, IBinaryOp<long>
    {
        if (a.IsInt64 != b.IsInt64)
            throw new NotSupportedException(
                $"{op} needs both operands float32 or both int64, got {(a.IsInt64 ? "int64" : "float32")} {a.ShapeString} and {(b.IsInt64 ? "int64" : "float32")} {b.ShapeString}; ONNX mixes element types only through an explicit Cast.");
        return a.IsInt64
            ? BinaryCore<long, TL>(a, b, a.I64!, b.I64!, op)
            : BinaryCore<float, TF>(a, b, a.Data, b.Data, op);
    }

    /// <summary>Allocates the engine's result tensor for its element type: float[]
    /// Data for <see cref="float"/>, long[] I64 for <see cref="long"/>. The typeof
    /// test is a JIT-time constant per value-type instantiation, so the specialised
    /// code carries no branch.</summary>
    private static Tensor NewNumeric<T>(int[] shape, out T[] payload)
    {
        Tensor t = typeof(T) == typeof(long)
            ? new Tensor(shape, new long[Tensor.CheckedProduct(shape)])
            : new Tensor(shape);
        payload = (T[])PayloadArray(t);
        return t;
    }

    /// <summary>The one binary engine behind Add/Sub/Mul/Div/Pow/PRelu, generic over
    /// the element type (float or long): a flat loop when the shapes are identical, a
    /// strided broadcast walk otherwise; both go parallel once the tensor is big
    /// enough (int64 operands are tiny mask/shape tensors and stay sequential).</summary>
    private static Tensor BinaryCore<T, TOp>(Tensor a, Tensor b, T[] xa, T[] xb, string op)
        where TOp : struct, IBinaryOp<T>
    {
        TOp f = default;

        // Fast path: identical shapes → one flat loop, no index arithmetic at all.
        if (a.Shape.AsSpan().SequenceEqual(b.Shape))
        {
            var y = NewNumeric<T>(a.Shape, out T[] yd);
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
        var result = NewNumeric<T>(outShape, out T[] rd);
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

    /// <summary>The unary engine behind Sigmoid/Relu/Exp/Sqrt/Erf/Clip: one flat loop,
    /// split into parallel blocks for big tensors. Stateless functors omit
    /// <paramref name="f"/>; Clip passes an instance carrying its bounds.</summary>
    private static Tensor Map<TOp>(Tensor x, string op, TOp f = default) where TOp : struct, IUnaryOp
    {
        RequireFloat(x, op);
        var y = new Tensor(x.Shape);
        float[] src = x.Data, dst = y.Data;
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
    private interface IBinaryOp<T> { T Apply(T a, T b); }

    private readonly struct SigmoidF : IUnaryOp { public float Apply(float v) => 1f / (1f + MathF.Exp(-v)); }
    private readonly struct ReluF : IUnaryOp { public float Apply(float v) => MathF.Max(v, 0f); }
    private readonly struct ExpF : IUnaryOp { public float Apply(float v) => MathF.Exp(v); }
    private readonly struct SqrtF : IUnaryOp { public float Apply(float v) => MathF.Sqrt(v); }
    private readonly struct SquareF : IUnaryOp { public float Apply(float v) => v * v; }
    private readonly struct ErfF : IUnaryOp { public float Apply(float v) => (float)ErfDouble(v); }

    /// <summary>Clamp functor: min first, then max, so max wins when min &gt; max
    /// (which is also why this cannot be Math.Clamp — that throws on min &gt; max).</summary>
    private readonly struct ClipF : IUnaryOp
    {
        private readonly float _lo, _hi;
        public ClipF(float lo, float hi) { _lo = lo; _hi = hi; }
        public float Apply(float v) => MathF.Min(MathF.Max(v, _lo), _hi);
    }

    private readonly struct AddF : IBinaryOp<float> { public float Apply(float a, float b) => a + b; }
    private readonly struct SubF : IBinaryOp<float> { public float Apply(float a, float b) => a - b; }
    private readonly struct MulF : IBinaryOp<float> { public float Apply(float a, float b) => a * b; }
    private readonly struct DivF : IBinaryOp<float> { public float Apply(float a, float b) => a / b; }
    private readonly struct PowF : IBinaryOp<float> { public float Apply(float a, float b) => MathF.Pow(a, b); }
    private readonly struct PReluF : IBinaryOp<float> { public float Apply(float v, float s) => v < 0f ? s * v : v; }

    private readonly struct AddL : IBinaryOp<long> { public long Apply(long a, long b) => a + b; }
    private readonly struct SubL : IBinaryOp<long> { public long Apply(long a, long b) => a - b; }
    private readonly struct MulL : IBinaryOp<long> { public long Apply(long a, long b) => a * b; }
    private readonly struct DivL : IBinaryOp<long> { public long Apply(long a, long b) => a / b; }   // truncates toward zero, per ONNX integer Div

    // ── scalar math helpers ──────────────────────────────────────────────────

    private const double TwoOverSqrtPi = 1.1283791670955126;   // 2/√π
    private const double SqrtPi = 1.7724538509055160;          // √π

    /// <summary>The error function in double precision, absolute error &lt; 1e-9 over
    /// all reals. Odd symmetry is exact by construction: the value is computed on |x|
    /// and the sign of x is applied at the end. |x| &lt; 3.75 uses the all-positive
    /// series erf(x) = 2/√π · e^(−x²) · Σ x·(2x²)^n / (1·3·5⋯(2n+1)) — no term
    /// cancellation, unlike the alternating Maclaurin series. Larger |x| goes through
    /// 1 − erfc(x) with erfc from the Abramowitz–Stegun 7.1.14 continued fraction
    /// √π·e^(x²)·erfc(x) = 1/(x + ½/(x + 1/(x + 3⁄2/(x + …)))), evaluated backward
    /// from depth 60. |x| ≥ 6 returns ±1 exactly: erfc(6) ≈ 2e-17 sits below one
    /// double ulp of 1.</summary>
    private static double ErfDouble(double x)
    {
        double ax = Math.Abs(x);
        if (ax >= 6.0) return x < 0.0 ? -1.0 : 1.0;

        double r;
        if (ax < 3.75)
        {
            double x2 = ax * ax;
            double term = ax;                        // n = 0 term: x / 1
            double sum = ax;
            for (int n = 1; n < 200; n++)
            {
                term *= 2.0 * x2 / (2 * n + 1);      // ratio of consecutive terms
                sum += term;
                if (term <= sum * 1e-17) break;      // below one double ulp of the sum
            }
            r = TwoOverSqrtPi * Math.Exp(-x2) * sum;
        }
        else
        {
            double cf = ax;
            for (int n = 60; n >= 1; n--)            // backward: deepest level first
                cf = ax + 0.5 * n / cf;
            r = 1.0 - Math.Exp(-ax * ax) / (SqrtPi * cf);
        }
        return x < 0.0 ? -r : r;
    }
}
