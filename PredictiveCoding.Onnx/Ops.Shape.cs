namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// Ops.Shape — shape-manipulation kernels: Reshape, Flatten, Transpose, Concat,
// Slice, Shape, Unsqueeze, Squeeze.
//
// Every kernel here works for BOTH tensor payload kinds — float (Data) and
// int64 (I64) — because ONNX shape pipelines (Shape → Slice → Concat → Reshape)
// flow int64 tensors through the same ops as the float feature maps. Pure
// re-labelling ops (Reshape/Flatten/Unsqueeze/Squeeze) share the payload array
// with the input since layout is row-major; the rest copy through a small set
// of payload-agnostic helpers at the bottom of the file.
// ─────────────────────────────────────────────────────────────────────────────

public static partial class Ops
{
    /// <summary>Reshape (opset 5+): re-labels <paramref name="x"/> to the dims given by the
    /// 1-D int64 <paramref name="shape"/> tensor. A 0 copies the corresponding input dim
    /// (allowzero=0 default); a single -1 is inferred from the element count. The payload
    /// array is shared with the input — row-major layout makes reshape free.</summary>
    public static Tensor Reshape(Tensor x, Tensor shape)
    {
        long[] requested = RequireI64(shape, "Reshape 'shape' input");

        var dims = new int[requested.Length];
        int inferAt = -1;
        long known = 1;
        for (int i = 0; i < requested.Length; i++)
        {
            long v = requested[i];
            if (v == -1)
            {
                if (inferAt >= 0)
                    throw new ArgumentException("Reshape allows at most one -1 in the target shape.");
                inferAt = i;
            }
            else if (v == 0)
            {
                if (i >= x.Rank)
                    throw new ArgumentException($"Reshape dim {i} is 0 but input rank is only {x.Rank}.");
                dims[i] = x.Shape[i];
                known *= dims[i];
            }
            else if (v < 0)
            {
                throw new ArgumentException($"Reshape dim {i} is {v}; only -1, 0 or positive values are allowed.");
            }
            else
            {
                dims[i] = checked((int)v);
                known *= v;
            }
        }

        if (inferAt >= 0)
        {
            if (known == 0 || x.Count % known != 0)
                throw new ArgumentException($"Reshape cannot infer -1: element count {x.Count} is not divisible by {known}.");
            dims[inferAt] = (int)(x.Count / known);
        }

        if (Tensor.CheckedProduct(dims) != x.Count)
            throw new ArgumentException($"Reshape target [{string.Join(",", dims)}] has a different element count than input {x.ShapeString}.");

        return WithShape(x, dims);
    }

    /// <summary>Flatten (opset 1+): collapses the input into a 2-D tensor
    /// [prod(dims &lt; axis), prod(dims ≥ axis)]; axis 0 gives [1, total].
    /// The payload array is shared with the input.</summary>
    public static Tensor Flatten(Tensor x, int axis)
    {
        int rank = x.Rank;
        if (axis < -rank || axis > rank)
            throw new ArgumentException($"Flatten axis {axis} out of range for rank {rank} (allowed [{-rank},{rank}]).");
        if (axis < 0) axis += rank;

        int outer = 1;
        for (int i = 0; i < axis; i++) outer *= x.Shape[i];
        int inner = 1;
        for (int i = axis; i < rank; i++) inner *= x.Shape[i];

        return WithShape(x, [outer, inner]);
    }

    /// <summary>Transpose (opset 1+): permutes the dimensions. An empty
    /// <paramref name="perm"/> means "reverse the dimension order" per the spec.
    /// Elements are copied via output-index → input-offset stride mapping.</summary>
    public static Tensor Transpose(Tensor x, long[] perm)
    {
        int rank = x.Rank;

        var p = new int[rank];
        if (perm.Length == 0)
        {
            for (int i = 0; i < rank; i++) p[i] = rank - 1 - i;
        }
        else
        {
            if (perm.Length != rank)
                throw new ArgumentException($"Transpose perm has {perm.Length} entries for a rank-{rank} tensor.");
            var seen = new bool[rank];
            for (int i = 0; i < rank; i++)
            {
                long v = perm[i];
                if (v < 0) v += rank;
                if (v < 0 || v >= rank)
                    throw new ArgumentException($"Transpose perm entry {perm[i]} out of range for rank {rank}.");
                if (seen[(int)v])
                    throw new ArgumentException($"Transpose perm [{string.Join(",", perm)}] is not a permutation: axis {v} repeats.");
                seen[(int)v] = true;
                p[i] = (int)v;
            }
        }

        var srcStrides = x.Strides();
        var outShape = new int[rank];
        var strides = new long[rank];
        for (int i = 0; i < rank; i++)
        {
            outShape[i] = x.Shape[p[i]];
            strides[i] = srcStrides[p[i]];
        }

        return GatherStrided(x, outShape, 0, strides);
    }

    /// <summary>Concat (opset 4+): joins tensors along <paramref name="axis"/> (negative
    /// counts from the end). All inputs must share rank, payload kind and every dim
    /// except the concat axis. Copies contiguous blocks, one per outer index per input.</summary>
    public static Tensor Concat(Tensor[] inputs, int axis)
    {
        if (inputs.Length == 0)
            throw new ArgumentException("Concat needs at least one input.");

        var first = inputs[0];
        int rank = first.Rank;
        if (axis < 0) axis += rank;
        if (axis < 0 || axis >= rank)
            throw new ArgumentException($"Concat axis out of range for rank {rank}.");

        int axisSum = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            var t = inputs[i];
            if (t.IsInt64 != first.IsInt64)
                throw new ArgumentException($"Concat input {i} payload kind differs from input 0 (float vs int64).");
            if (t.Rank != rank)
                throw new ArgumentException($"Concat input {i} has rank {t.Rank}, expected {rank}.");
            for (int d = 0; d < rank; d++)
                if (d != axis && t.Shape[d] != first.Shape[d])
                    throw new ArgumentException($"Concat input {i} shape {t.ShapeString} mismatches {first.ShapeString} on non-concat dim {d}.");
            axisSum += t.Shape[axis];
        }

        var outShape = (int[])first.Shape.Clone();
        outShape[axis] = axisSum;

        int inner = 1;                               // elements per step along the concat axis
        for (int d = axis + 1; d < rank; d++) inner *= outShape[d];
        int outer = 1;                               // independent blocks before the concat axis
        for (int d = 0; d < axis; d++) outer *= outShape[d];

        var result = NewLike(first, outShape);
        Array dst = PayloadArray(result);
        int dstBlock = axisSum * inner;

        int axisOffset = 0;
        foreach (var t in inputs)
        {
            Array src = PayloadArray(t);
            int block = t.Shape[axis] * inner;
            for (int o = 0; o < outer; o++)
                Array.Copy(src, o * block, dst, o * dstBlock + axisOffset * inner, block);
            axisOffset += t.Shape[axis];
        }
        return result;
    }

    /// <summary>Slice (opset 10+, input form): extracts [start:end:step] per axis.
    /// <paramref name="starts"/>/<paramref name="ends"/> are required 1-D int64 tensors;
    /// <paramref name="axes"/> defaults to 0..k-1 and <paramref name="steps"/> to all 1s.
    /// Negative starts/ends/axes count from the end; out-of-range values are clamped
    /// per the spec. Unlisted axes pass through whole.</summary>
    public static Tensor Slice(Tensor data, Tensor starts, Tensor ends, Tensor? axes, Tensor? steps)
    {
        long[] startVals = RequireI64(starts, "Slice 'starts' input");
        long[] endVals = RequireI64(ends, "Slice 'ends' input");
        if (startVals.Length != endVals.Length)
            throw new ArgumentException($"Slice starts ({startVals.Length}) and ends ({endVals.Length}) lengths differ.");
        int k = startVals.Length;

        long[]? axisVals = axes is null ? null : RequireI64(axes, "Slice 'axes' input");
        if (axisVals is not null && axisVals.Length != k)
            throw new ArgumentException($"Slice axes has {axisVals.Length} entries, expected {k}.");
        long[]? stepVals = steps is null ? null : RequireI64(steps, "Slice 'steps' input");
        if (stepVals is not null && stepVals.Length != k)
            throw new ArgumentException($"Slice steps has {stepVals.Length} entries, expected {k}.");

        int rank = data.Rank;
        var start = new long[rank];                  // resolved per-axis window
        var step = new long[rank];
        var outShape = new int[rank];
        for (int ax = 0; ax < rank; ax++)
        {
            step[ax] = 1;
            outShape[ax] = data.Shape[ax];           // unlisted axes pass through whole
        }

        var listed = new bool[rank];
        for (int i = 0; i < k; i++)
        {
            long axL = axisVals is not null ? axisVals[i] : i;
            if (axL < 0) axL += rank;
            if (axL < 0 || axL >= rank)
                throw new ArgumentException($"Slice axis {(axisVals is not null ? axisVals[i] : i)} out of range for rank {rank}.");
            int ax = (int)axL;
            if (listed[ax])
                throw new ArgumentException($"Slice lists axis {ax} more than once.");
            listed[ax] = true;

            long s = stepVals is not null ? stepVals[i] : 1;
            if (s == 0)
                throw new ArgumentException("Slice step must be non-zero.");
            step[ax] = s;

            long d = data.Shape[ax];
            if (d == 0) { outShape[ax] = 0; continue; }

            long b = startVals[i];
            if (b < 0) b += d;
            long e = endVals[i];
            if (e < 0) e += d;
            if (s > 0)
            {
                b = Math.Clamp(b, 0, d);
                e = Math.Clamp(e, 0, d);
            }
            else
            {
                b = Math.Clamp(b, 0, d - 1);
                e = Math.Clamp(e, -1, d - 1);
            }

            start[ax] = b;
            long n = (e - b + (s > 0 ? s - 1 : s + 1)) / s;   // ceil((end-start)/step)
            outShape[ax] = (int)Math.Max(0, n);
        }

        var srcStrides = data.Strides();
        long baseOffset = 0;
        var strides = new long[rank];
        for (int ax = 0; ax < rank; ax++)
        {
            baseOffset += start[ax] * srcStrides[ax];
            strides[ax] = step[ax] * srcStrides[ax];
        }

        return GatherStrided(data, outShape, baseOffset, strides);
    }

    /// <summary>Shape (opset 1+): returns the input's dims as a 1-D int64 tensor of
    /// length rank.</summary>
    public static Tensor Shape(Tensor x)
    {
        var dims = new long[x.Rank];
        for (int i = 0; i < x.Rank; i++) dims[i] = x.Shape[i];
        return new Tensor([x.Rank], dims);
    }

    /// <summary>Unsqueeze: inserts 1-dims at the given axes. Axes come from
    /// <paramref name="axesTensor"/> (int64, opset ≥ 13) when present, else the 'axes'
    /// attribute (opset ≤ 12); negative values count from the OUTPUT rank
    /// (rank + len(axes)). The payload array is shared with the input.</summary>
    public static Tensor Unsqueeze(Tensor x, Tensor? axesTensor, long[] axesAttr)
    {
        long[] axes = axesTensor is not null ? RequireI64(axesTensor, "Unsqueeze 'axes' input") : axesAttr;
        if (axes.Length == 0)
            throw new ArgumentException("Unsqueeze requires axes (tensor input or attribute).");

        int outRank = x.Rank + axes.Length;
        var insertAt = new bool[outRank];
        foreach (long a in axes)
        {
            long v = a;
            if (v < 0) v += outRank;
            if (v < 0 || v >= outRank)
                throw new ArgumentException($"Unsqueeze axis {a} out of range for output rank {outRank}.");
            if (insertAt[(int)v])
                throw new ArgumentException($"Unsqueeze lists axis {v} more than once.");
            insertAt[(int)v] = true;
        }

        var outShape = new int[outRank];
        int src = 0;
        for (int i = 0; i < outRank; i++)
            outShape[i] = insertAt[i] ? 1 : x.Shape[src++];

        return WithShape(x, outShape);
    }

    /// <summary>Squeeze: removes 1-dims. With axes (tensor input, opset ≥ 13, or the
    /// 'axes' attribute) only those dims are removed and each must be 1; without any
    /// axes every dim equal to 1 is removed. The payload array is shared with the
    /// input.</summary>
    public static Tensor Squeeze(Tensor x, Tensor? axesTensor, long[] axesAttr)
    {
        long[]? axes = axesTensor is not null ? RequireI64(axesTensor, "Squeeze 'axes' input")
                     : axesAttr.Length > 0 ? axesAttr
                     : null;

        int rank = x.Rank;
        var remove = new bool[rank];
        if (axes is null)
        {
            for (int i = 0; i < rank; i++) remove[i] = x.Shape[i] == 1;
        }
        else
        {
            foreach (long a in axes)
            {
                long v = a;
                if (v < 0) v += rank;
                if (v < 0 || v >= rank)
                    throw new ArgumentException($"Squeeze axis {a} out of range for rank {rank}.");
                if (x.Shape[(int)v] != 1)
                    throw new ArgumentException($"Squeeze axis {v} has size {x.Shape[(int)v]}; only 1-dims can be squeezed.");
                remove[(int)v] = true;
            }
        }

        var outShape = new List<int>(rank);
        for (int i = 0; i < rank; i++)
            if (!remove[i]) outShape.Add(x.Shape[i]);

        return WithShape(x, [.. outShape]);
    }

    // ── payload-agnostic helpers ─────────────────────────────────────────────

    /// <summary>Re-labels <paramref name="x"/> with a new shape, sharing its payload
    /// array (valid because layout is row-major and element count is unchanged).</summary>
    private static Tensor WithShape(Tensor x, int[] shape)
        => x.IsInt64 ? new Tensor(shape, x.I64!) : new Tensor(shape, x.Data);

    /// <summary>Allocates a tensor of the given shape with the same payload kind
    /// (float or int64) as <paramref name="like"/>.</summary>
    private static Tensor NewLike(Tensor like, int[] shape)
        => like.IsInt64 ? new Tensor(shape, new long[Tensor.CheckedProduct(shape)]) : new Tensor(shape);

    /// <summary>The backing array (float[] or long[]) as a non-generic Array, so block
    /// copies work for either payload kind via <see cref="Array.Copy(Array,int,Array,int,int)"/>.</summary>
    private static Array PayloadArray(Tensor t) => t.IsInt64 ? t.I64! : t.Data;

    /// <summary>Int64 payload of a parameter tensor, with a clear error when a graph
    /// wired a float tensor into a shape/index slot.</summary>
    private static long[] RequireI64(Tensor t, string what)
        => t.I64 ?? throw new ArgumentException($"{what} must be an int64 tensor, got float {t.ShapeString}.");

    /// <summary>Copies into a fresh same-kind tensor of <paramref name="outShape"/>,
    /// reading each output element from the source at
    /// baseOffset + Σ index[i]·strides[i] — the shared engine behind Transpose
    /// (base 0, permuted strides) and Slice (start offsets, stepped strides).</summary>
    private static Tensor GatherStrided(Tensor src, int[] outShape, long baseOffset, long[] strides)
    {
        var result = NewLike(src, outShape);
        if (src.IsInt64) GatherStridedCore(src.I64!, result.I64!, outShape, baseOffset, strides);
        else GatherStridedCore(src.Data, result.Data, outShape, baseOffset, strides);
        return result;
    }

    private static void GatherStridedCore<T>(T[] src, T[] dst, int[] outShape, long baseOffset, long[] strides)
    {
        if (dst.Length == 0) return;
        int rank = outShape.Length;
        if (rank == 0) { dst[0] = src[(int)baseOffset]; return; }

        int rowLen = outShape[rank - 1];             // innermost (last-axis) run
        long rowStride = strides[rank - 1];
        int rows = dst.Length / rowLen;

        var idx = new int[rank - 1];                 // multi-index over the outer axes
        long rowBase = baseOffset;
        int dstPos = 0;
        for (int r = 0; r < rows; r++)
        {
            if (rowStride == 1)                      // contiguous run → block copy
            {
                Array.Copy(src, (int)rowBase, dst, dstPos, rowLen);
            }
            else
            {
                long off = rowBase;
                for (int c = 0; c < rowLen; c++, off += rowStride)
                    dst[dstPos + c] = src[(int)off];
            }
            dstPos += rowLen;

            for (int ax = rank - 2; ax >= 0; ax--)   // odometer-increment the outer index
            {
                idx[ax]++;
                rowBase += strides[ax];
                if (idx[ax] < outShape[ax]) break;
                rowBase -= (long)idx[ax] * strides[ax];
                idx[ax] = 0;
            }
        }
    }
}
