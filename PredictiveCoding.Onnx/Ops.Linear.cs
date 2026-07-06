namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// Ops.Linear — dense linear algebra and normalisation: Gemm, MatMul, BatchNorm.
//
// Gemm and MatMul run the same cache-friendly i,k,j loop: the inner j-loop
// streams one output row against one B row — both contiguous spans — so it
// vectorises and prefetches well. Transposed Gemm operands are materialised
// into row-major copies once up front instead of paying strided reads M times
// over. MatMul broadcasts leading batch dims numpy-style and reuses the 2-D
// core per batch matrix. BatchNorm (inference form) folds scale, mean,
// variance and bias into one per-channel multiply-add, a single linear pass.
// ─────────────────────────────────────────────────────────────────────────────

public static partial class Ops
{
    /// <summary>Work size (M·N·K) above which the matrix kernels parallelise over
    /// output rows or batch matrices.</summary>
    private const long MatParallelCutoff = 1_000_000;

    /// <summary>Gemm (opset 7+): Y = alpha·A′·B′ + beta·C with A′ = (transA ? Aᵀ : A)
    /// of shape [M,K] and B′ = (transB ? Bᵀ : B) of shape [K,N]. 2-D only. C — when
    /// present and beta ≠ 0 — broadcasts unidirectionally to [M,N]: [N], [1,N], [M,1],
    /// [M,N] or a scalar.</summary>
    public static Tensor Gemm(Tensor a, Tensor b, Tensor? c, float alpha, float beta, bool transA, bool transB)
    {
        RequireFloat(a, "Gemm");
        RequireFloat(b, "Gemm");
        if (c is not null) RequireFloat(c, "Gemm");
        if (a.Rank != 2 || b.Rank != 2)
            throw new NotSupportedException($"Gemm is 2-D only: got A {a.ShapeString}, B {b.ShapeString}.");

        int M = transA ? a.Shape[1] : a.Shape[0];
        int K = transA ? a.Shape[0] : a.Shape[1];
        int kB = transB ? b.Shape[1] : b.Shape[0];
        int N = transB ? b.Shape[0] : b.Shape[1];
        if (K != kB)
            throw new ArgumentException(
                $"Gemm inner dimensions disagree: A {a.ShapeString} (transA={transA}) × B {b.ShapeString} (transB={transB}).");

        // Row-major [M,K] / [K,N] views: a transposed operand is copied once here so
        // the hot loop below only ever streams contiguous rows.
        float[] ad = transA ? Transpose2D(a.Data, a.Shape[0], a.Shape[1]) : a.Data;
        float[] bd = transB ? Transpose2D(b.Data, b.Shape[0], b.Shape[1]) : b.Data;

        // Resolve C's unidirectional broadcast as a (row stride, column stride) pair
        // over [M,N] — 0 along a stretched axis, so the same value is re-read.
        float[]? cd = null;
        int cRow = 0, cCol = 0;
        if (c is not null && beta != 0f)
        {
            cd = c.Data;
            switch (c.Rank)
            {
                case 0:                              // rank-0 scalar: both strides stay 0
                    break;
                case 1 when c.Shape[0] == N:
                    cCol = 1;
                    break;
                case 1 when c.Shape[0] == 1:         // scalar as [1]
                    break;
                case 2 when (c.Shape[0] == M || c.Shape[0] == 1) && (c.Shape[1] == N || c.Shape[1] == 1):
                    cRow = c.Shape[0] == 1 ? 0 : c.Shape[1];
                    cCol = c.Shape[1] == 1 ? 0 : 1;
                    break;
                default:
                    throw new ArgumentException($"Gemm C {c.ShapeString} does not broadcast to [{M},{N}].");
            }
        }

        var y = new Tensor([M, N]);
        float[] yd = y.Data;

        void DoRow(int i)
        {
            var yRow = yd.AsSpan(i * N, N);
            if (cd is not null)
                for (int j = 0, cp = i * cRow; j < N; j++, cp += cCol)
                    yRow[j] = beta * cd[cp];

            int aRow = i * K;
            for (int k = 0; k < K; k++)
            {
                float aik = alpha * ad[aRow + k];
                var bRow = bd.AsSpan(k * N, N);
                for (int j = 0; j < N; j++)
                    yRow[j] += aik * bRow[j];
            }
        }

        if ((long)M * N * K > MatParallelCutoff && M > 1) Parallel.For(0, M, DoRow);
        else for (int i = 0; i < M; i++) DoRow(i);
        return y;
    }

    /// <summary>MatMul (opset 1+): matrix product over the last two axes; leading
    /// batch axes broadcast numpy-style. Rank must be ≥ 2 on both sides — the 1-D
    /// vector forms of the spec are not needed by our models.</summary>
    public static Tensor MatMul(Tensor a, Tensor b)
    {
        RequireFloat(a, "MatMul");
        RequireFloat(b, "MatMul");
        if (a.Rank < 2 || b.Rank < 2)
            throw new NotSupportedException(
                $"MatMul supports rank ≥ 2 only (no 1-D vector operands): got {a.ShapeString} and {b.ShapeString}.");

        int M = a.Shape[^2], K = a.Shape[^1];
        int kB = b.Shape[^2], N = b.Shape[^1];
        if (K != kB)
            throw new ArgumentException($"MatMul inner dimensions disagree: {a.ShapeString} × {b.ShapeString}.");

        if (a.Rank == 2 && b.Rank == 2)              // the common case: one matrix product
        {
            var y2 = new Tensor([M, N]);
            MatMul2D(a.Data, 0, b.Data, 0, y2.Data, 0, M, K, N,
                     parallel: (long)M * N * K > MatParallelCutoff);
            return y2;
        }

        // Batched: broadcast the leading dims. Every matrix is a contiguous
        // [M,K] / [K,N] block in row-major order, so each batch coordinate reduces
        // to an (offset into a, offset into b) pair in whole-matrix units.
        int[] batchA = a.Shape[..^2];
        int[] batchB = b.Shape[..^2];
        int[] batch = BroadcastShape(batchA, batchB, "MatMul");
        int batchCount = Tensor.CheckedProduct(batch);

        var outShape = new int[batch.Length + 2];
        Array.Copy(batch, outShape, batch.Length);
        outShape[^2] = M;
        outShape[^1] = N;
        var y = new Tensor(outShape);

        int[] sa = BatchStrides(batchA, batch.Length);
        int[] sb = BatchStrides(batchB, batch.Length);
        int aMat = M * K, bMat = K * N, yMat = M * N;
        float[] ad = a.Data, bd = b.Data, yd = y.Data;

        void DoBatch(int bi)
        {
            int offA = 0, offB = 0;                  // decompose bi into batch coords
            for (int ax = batch.Length - 1, rem = bi; ax >= 0; ax--)
            {
                int idx = rem % batch[ax];
                rem /= batch[ax];
                offA += idx * sa[ax];
                offB += idx * sb[ax];
            }
            MatMul2D(ad, offA * aMat, bd, offB * bMat, yd, bi * yMat, M, K, N, parallel: false);
        }

        if ((long)batchCount * M * N * K > MatParallelCutoff && batchCount > 1)
            Parallel.For(0, batchCount, DoBatch);
        else
            for (int bi = 0; bi < batchCount; bi++) DoBatch(bi);
        return y;
    }

    /// <summary>BatchNormalization (inference form, opset 7+): per-channel (axis 1)
    /// y = scale[c]·(x − mean[c]) / √(var[c] + eps) + bias[c], folded into one
    /// multiply-add per element: y = a[c]·x + b[c] with a = scale/√(var+eps) and
    /// b = bias − a·mean.</summary>
    public static Tensor BatchNorm(Tensor x, Tensor scale, Tensor bias, Tensor mean, Tensor variance, float eps)
    {
        RequireFloat(x, "BatchNormalization");
        RequireFloat(scale, "BatchNormalization");
        RequireFloat(bias, "BatchNormalization");
        RequireFloat(mean, "BatchNormalization");
        RequireFloat(variance, "BatchNormalization");
        if (x.Rank < 2)
            throw new NotSupportedException(
                $"BatchNormalization needs a channel axis: input rank must be ≥ 2, got {x.ShapeString}.");

        int C = x.Shape[1];
        CheckChannelParam(scale, C, "scale");
        CheckChannelParam(bias, C, "bias");
        CheckChannelParam(mean, C, "mean");
        CheckChannelParam(variance, C, "variance");

        var mulC = new float[C];                     // a[c] = scale/√(var+eps)
        var addC = new float[C];                     // b[c] = bias − a·mean
        for (int ch = 0; ch < C; ch++)
        {
            float aVal = scale.Data[ch] / MathF.Sqrt(variance.Data[ch] + eps);
            mulC[ch] = aVal;
            addC[ch] = bias.Data[ch] - aVal * mean.Data[ch];
        }

        int spatial = 1;                             // elements per channel plane
        for (int d = 2; d < x.Rank; d++) spatial *= x.Shape[d];
        int planes = x.Shape[0] * C;                 // N·C channel planes, each contiguous

        var y = new Tensor(x.Shape);
        float[] xd = x.Data, yd = y.Data;

        void DoPlane(int p)
        {
            int ch = p % C;
            float aVal = mulC[ch], bVal = addC[ch];
            int off = p * spatial;
            for (int i = 0; i < spatial; i++)
                yd[off + i] = aVal * xd[off + i] + bVal;
        }

        if (xd.Length > ParallelCutoff && planes > 1) Parallel.For(0, planes, DoPlane);
        else for (int p = 0; p < planes; p++) DoPlane(p);
        return y;
    }

    // ── shared matrix plumbing ───────────────────────────────────────────────

    /// <summary>Row-major [m,k]·[k,n] product into <paramref name="dst"/> (assumed
    /// zeroed) at the given offsets — the shared core of MatMul's 2-D fast path and
    /// its batched loop. i,k,j order: the j-loop streams contiguous rows of both the
    /// output and b.</summary>
    private static void MatMul2D(float[] a, int aOff, float[] b, int bOff, float[] dst, int dstOff,
                                 int m, int k, int n, bool parallel)
    {
        void DoRow(int i)
        {
            var yRow = dst.AsSpan(dstOff + i * n, n);
            int aRow = aOff + i * k;
            for (int kk = 0; kk < k; kk++)
            {
                float aik = a[aRow + kk];
                var bRow = b.AsSpan(bOff + kk * n, n);
                for (int j = 0; j < n; j++)
                    yRow[j] += aik * bRow[j];
            }
        }

        if (parallel && m > 1) Parallel.For(0, m, DoRow);
        else for (int i = 0; i < m; i++) DoRow(i);
    }

    /// <summary>Dense 2-D transpose: [rows,cols] → [cols,rows]. Used to normalise
    /// Gemm's transA/transB operands into row-major form once, up front.</summary>
    private static float[] Transpose2D(float[] src, int rows, int cols)
    {
        var dst = new float[src.Length];
        for (int r = 0; r < rows; r++)
        {
            int srcRow = r * cols;
            for (int cIdx = 0; cIdx < cols; cIdx++)
                dst[cIdx * rows + r] = src[srcRow + cIdx];
        }
        return dst;
    }

    /// <summary>Strides — in whole matrices — for locating one MatMul input's matrix
    /// block from a broadcast batch coordinate: 0 along stretched 1-dims and missing
    /// leading dims, the input's own row-major batch stride elsewhere.</summary>
    private static int[] BatchStrides(int[] batchDims, int outRank)
    {
        var strides = new int[outRank];
        int acc = 1;
        int shift = outRank - batchDims.Length;
        for (int i = batchDims.Length - 1; i >= 0; i--)
        {
            if (batchDims[i] != 1) strides[shift + i] = acc;
            acc *= batchDims[i];
        }
        return strides;
    }

    /// <summary>Validates that a per-channel BatchNormalization parameter has exactly
    /// one value per channel.</summary>
    private static void CheckChannelParam(Tensor t, int channels, string name)
    {
        if (t.Count != channels)
            throw new ArgumentException(
                $"BatchNormalization '{name}' {t.ShapeString} must have {channels} values (one per channel).");
    }
}
