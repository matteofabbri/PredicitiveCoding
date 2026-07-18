namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// Ops.Reduce — reduction and scan kernels: ReduceMean, ReduceL2, CumSum.
//
// ReduceMean and ReduceL2 share one engine. Axes resolve from the int64 tensor
// input (opset ≥ 18) or the 'axes' attribute (opset ≤ 17 — the arctic-embed
// path); no axes at all means "reduce everything" unless noop_with_empty_axes
// asks for a pass-through. Each output element then sums its reduced index
// space through precomputed strides into a DOUBLE accumulator — LayerNorm
// feeds 1024-wide rows and float accumulation would cost parity vs ONNX
// Runtime. When the reduced axes form a trailing block (the LayerNorm
// last-axis case) every output element is one contiguous run, so a fast path
// sums sequential rows and parallelises over them past the size cutoff.
//
// CumSum is a sequential scan along a single axis, composed per the spec from
// 'exclusive' (sum of strictly-preceding) and 'reverse' (scan from the end).
// It works for BOTH payload kinds: double accumulator per run for float,
// long for int64.
// ─────────────────────────────────────────────────────────────────────────────

public static partial class Ops
{
    /// <summary>ReduceMean (opset 13-18): mean over the reduced axes. Axes come from
    /// <paramref name="axesTensor"/> (1-D int64, opset ≥ 18) when present, else the
    /// 'axes' attribute (opset ≤ 17); negative axes count from the end. With no axes
    /// at all the whole tensor reduces to one value, unless
    /// <paramref name="noopWithEmptyAxes"/> is set — then the input passes through
    /// unchanged. <paramref name="keepDims"/> keeps the reduced dims as 1-dims.
    /// Float only; each output element accumulates in double.</summary>
    public static Tensor ReduceMean(Tensor x, Tensor? axesTensor, long[] axesAttr, bool keepDims, bool noopWithEmptyAxes)
        => Reduce(x, axesTensor, axesAttr, keepDims, noopWithEmptyAxes, l2: false, "ReduceMean");

    /// <summary>ReduceL2 (opset 13-18): √(Σ v²) over the reduced axes — the final
    /// sentence-embedding L2 norm. Same axes/keepdims/noop semantics as
    /// <see cref="ReduceMean"/>. Float only; each output element accumulates in
    /// double.</summary>
    public static Tensor ReduceL2(Tensor x, Tensor? axesTensor, long[] axesAttr, bool keepDims, bool noopWithEmptyAxes)
        => Reduce(x, axesTensor, axesAttr, keepDims, noopWithEmptyAxes, l2: true, "ReduceL2");

    /// <summary>CumSum (opset 11+): running sum along <paramref name="axis"/> (a
    /// one-element int64 tensor; negative counts from the end). Works for both payload
    /// kinds — float lines accumulate in double, int64 lines in long.
    /// <paramref name="exclusive"/> makes each element the sum of STRICTLY-preceding
    /// elements (first = 0); <paramref name="reverse"/> runs the scan from the end
    /// toward the start (combined: sum of strictly-following).</summary>
    public static Tensor CumSum(Tensor x, Tensor axis, bool exclusive, bool reverse)
    {
        long[] axisVals = RequireI64(axis, "CumSum 'axis' input");
        if (axisVals.Length != 1)
            throw new ArgumentException($"CumSum 'axis' must hold exactly one value, got {axisVals.Length}.");

        int rank = x.Rank;
        long axL = axisVals[0];
        if (axL < 0) axL += rank;
        if (axL < 0 || axL >= rank)
            throw new ArgumentException($"CumSum axis {axisVals[0]} out of range for rank {rank}.");
        int ax = (int)axL;

        int axisLen = x.Shape[ax];
        int inner = 1;                               // elements per step along the axis
        for (int d = ax + 1; d < rank; d++) inner *= x.Shape[d];
        int outer = 1;                               // independent blocks before the axis
        for (int d = 0; d < ax; d++) outer *= x.Shape[d];

        var result = NewLike(x, (int[])x.Shape.Clone());
        if (x.IsInt64) CumSumCore(x.I64!, result.I64!, outer, axisLen, inner, exclusive, reverse);
        else CumSumCore(x.Data, result.Data, outer, axisLen, inner, exclusive, reverse);
        return result;
    }

    // ── shared reduction engine ──────────────────────────────────────────────

    /// <summary>Engine behind ReduceMean (mean of v) and ReduceL2 (√Σv²), selected by
    /// <paramref name="l2"/>. Resolves axes, builds the output shape, then either runs
    /// the contiguous-row fast path (reduced axes form a trailing block) or a generic
    /// stride walk over each output element's reduced index space.</summary>
    private static Tensor Reduce(Tensor x, Tensor? axesTensor, long[] axesAttr, bool keepDims, bool noopWithEmptyAxes, bool l2, string op)
    {
        RequireFloat(x, op);

        bool[]? reduceFlags = ResolveReduceAxes(x, axesTensor, axesAttr, noopWithEmptyAxes, op);
        if (reduceFlags is null)                     // noop_with_empty_axes pass-through
            return WithShape(x, (int[])x.Shape.Clone());

        int rank = x.Rank;
        var srcStrides = x.Strides();

        int keptRank = 0, redRank = 0;
        for (int i = 0; i < rank; i++)
            if (reduceFlags[i]) redRank++; else keptRank++;

        var keptDims = new int[keptRank];            // outer (surviving) index space
        var keptStr = new int[keptRank];
        var redDims = new int[redRank];              // reduced (summed-away) index space
        var redStr = new int[redRank];
        var outShape = new List<int>(rank);
        long redCount = 1;
        for (int i = 0, k = 0, r = 0; i < rank; i++)
        {
            if (reduceFlags[i])
            {
                redDims[r] = x.Shape[i];
                redStr[r++] = srcStrides[i];
                redCount *= x.Shape[i];
                if (keepDims) outShape.Add(1);
            }
            else
            {
                keptDims[k] = x.Shape[i];
                keptStr[k++] = srcStrides[i];
                outShape.Add(x.Shape[i]);
            }
        }

        var result = new Tensor([.. outShape]);
        if (redCount == 0)                           // reduction over an empty set
        {
            Array.Fill(result.Data, l2 ? 0f : float.NaN);
            return result;
        }

        // Fast path: the reduced axes are exactly the LAST redRank axes, so every
        // output element is one contiguous row of the source (the LayerNorm case).
        int firstRed = rank - redRank;
        bool trailing = redRank > 0;
        for (int i = firstRed; i < rank && trailing; i++)
            trailing = reduceFlags[i];

        if (trailing) ReduceRows(x.Data, result.Data, (int)redCount, l2);
        else ReduceStrided(x.Data, result.Data, keptDims, keptStr, redDims, redStr, redCount, l2);
        return result;
    }

    /// <summary>Resolves the reduce-axes flags: <paramref name="axesTensor"/> (opset ≥ 18)
    /// wins when present, else <paramref name="axesAttr"/> (opset ≤ 17). Negative axes
    /// count from the end; duplicates throw. No axes at all reduces over EVERY axis,
    /// or — with <paramref name="noopWithEmptyAxes"/> — returns null, meaning "pass the
    /// input through unchanged".</summary>
    private static bool[]? ResolveReduceAxes(Tensor x, Tensor? axesTensor, long[] axesAttr, bool noopWithEmptyAxes, string op)
    {
        long[] axes = axesTensor is not null ? RequireI64(axesTensor, $"{op} 'axes' input") : axesAttr;

        int rank = x.Rank;
        var reduce = new bool[rank];
        if (axes.Length == 0)
        {
            if (noopWithEmptyAxes) return null;
            for (int i = 0; i < rank; i++) reduce[i] = true;
            return reduce;
        }

        foreach (long a in axes)
        {
            long v = a;
            if (v < 0) v += rank;
            if (v < 0 || v >= rank)
                throw new ArgumentException($"{op} axis {a} out of range for rank {rank}.");
            if (reduce[(int)v])
                throw new ArgumentException($"{op} lists axis {v} more than once.");
            reduce[(int)v] = true;
        }
        return reduce;
    }

    /// <summary>Contiguous fast path: each output element is one sequential source row
    /// of <paramref name="rowLen"/> elements. Rows run on the thread pool past the
    /// size cutoff — LayerNorm reduces [1,seq,1024] rows and dominates the encoder.</summary>
    private static void ReduceRows(float[] src, float[] dst, int rowLen, bool l2)
    {
        int rows = dst.Length;

        void DoRow(int r)
        {
            double acc = 0;
            int b = r * rowLen;
            if (l2)
            {
                for (int c = 0; c < rowLen; c++) { double v = src[b + c]; acc += v * v; }
                dst[r] = (float)Math.Sqrt(acc);
            }
            else
            {
                for (int c = 0; c < rowLen; c++) acc += src[b + c];
                dst[r] = (float)(acc / rowLen);
            }
        }

        if ((long)rows * rowLen > ParallelCutoff && rows > 1) Parallel.For(0, rows, DoRow);
        else for (int r = 0; r < rows; r++) DoRow(r);
    }

    /// <summary>Generic walk for non-contiguous reductions: an odometer over the kept
    /// axes picks each output element's base offset, an inner odometer over the reduced
    /// axes sums its index space through the source strides.</summary>
    private static void ReduceStrided(float[] src, float[] dst, int[] keptDims, int[] keptStr, int[] redDims, int[] redStr, long redCount, bool l2)
    {
        int keptRank = keptDims.Length;
        int redRank = redDims.Length;
        var keptIdx = new int[keptRank];
        var redIdx = new int[redRank];

        long baseOff = 0;
        for (int o = 0; o < dst.Length; o++)
        {
            double acc = 0;
            long off = baseOff;
            for (long r = 0; r < redCount; r++)
            {
                double v = src[(int)off];
                acc += l2 ? v * v : v;

                for (int ax = redRank - 1; ax >= 0; ax--)   // odometer over the reduced axes
                {
                    redIdx[ax]++;
                    off += redStr[ax];
                    if (redIdx[ax] < redDims[ax]) break;
                    off -= (long)redIdx[ax] * redStr[ax];
                    redIdx[ax] = 0;
                }
            }
            dst[o] = l2 ? (float)Math.Sqrt(acc) : (float)(acc / redCount);

            for (int ax = keptRank - 1; ax >= 0; ax--)      // odometer over the kept axes
            {
                keptIdx[ax]++;
                baseOff += keptStr[ax];
                if (keptIdx[ax] < keptDims[ax]) break;
                baseOff -= (long)keptIdx[ax] * keptStr[ax];
                keptIdx[ax] = 0;
            }
        }
    }

    // ── CumSum scan cores (one per payload kind) ─────────────────────────────

    /// <summary>Float scan: each line along the axis (outer × inner lines, stride =
    /// inner) accumulates sequentially in a DOUBLE, cast to float per element.</summary>
    private static void CumSumCore(float[] src, float[] dst, int outer, int axisLen, int inner, bool exclusive, bool reverse)
    {
        int start = reverse ? axisLen - 1 : 0;
        int step = reverse ? -inner : inner;

        for (int o = 0; o < outer; o++)
            for (int i = 0; i < inner; i++)
            {
                int pos = o * axisLen * inner + i + start * inner;
                double acc = 0;
                for (int k = 0; k < axisLen; k++, pos += step)
                {
                    if (exclusive) { dst[pos] = (float)acc; acc += src[pos]; }
                    else { acc += src[pos]; dst[pos] = (float)acc; }
                }
            }
    }

    /// <summary>Int64 scan: same walk with a LONG accumulator (exact; used by the
    /// attention-mask position pipeline, where CumSum builds position ids).</summary>
    private static void CumSumCore(long[] src, long[] dst, int outer, int axisLen, int inner, bool exclusive, bool reverse)
    {
        int start = reverse ? axisLen - 1 : 0;
        int step = reverse ? -inner : inner;

        for (int o = 0; o < outer; o++)
            for (int i = 0; i < inner; i++)
            {
                int pos = o * axisLen * inner + i + start * inner;
                long acc = 0;
                for (int k = 0; k < axisLen; k++, pos += step)
                {
                    if (exclusive) { dst[pos] = acc; acc += src[pos]; }
                    else { acc += src[pos]; dst[pos] = acc; }
                }
            }
    }
}
