namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// Ops.Pool — 2-D pooling kernels: MaxPool, AveragePool, GlobalAveragePool.
//
// All operate on NCHW float tensors and are parallelised over (batch × channel)
// planes. Padding is virtual: each window is clamped to the image, so max never
// sees a padded value (−infinity semantics) and average can count either the
// full kernel area or just the non-pad elements, per count_include_pad.
// ─────────────────────────────────────────────────────────────────────────────

public static partial class Ops
{
    /// <summary>
    /// 2-D max pooling over NCHW input. Padding behaves as −infinity, i.e. it never
    /// wins the max; a window that is entirely padding yields <see cref="float.NegativeInfinity"/>.
    /// Empty attribute arrays mean ONNX defaults: strides [1,1], pads [0,0,0,0]
    /// ([top, left, bottom, right]); <paramref name="kernel"/> is required.
    /// </summary>
    public static Tensor MaxPool(Tensor x, long[] kernel, long[] strides, long[] pads, bool ceilMode)
    {
        var (N, C, H, W, kH, kW, strideH, strideW, padT, padL, padB, padR) = PoolGeometry(x, kernel, strides, pads, "MaxPool");
        int outH = PoolOutDim(H, kH, strideH, padT, padB, ceilMode, "MaxPool");
        int outW = PoolOutDim(W, kW, strideW, padL, padR, ceilMode, "MaxPool");

        var y = new Tensor([N, C, outH, outW]);
        float[] xd = x.Data, yd = y.Data;
        int inPlane = H * W, outPlane = outH * outW;

        Parallel.For(0, N * C, nc =>
        {
            int xBase = nc * inPlane;
            int yBase = nc * outPlane;
            for (int oy = 0; oy < outH; oy++)
            {
                int iy0 = oy * strideH - padT;
                int yLo = Math.Max(0, iy0), yHi = Math.Min(H, iy0 + kH);
                int yRow = yBase + oy * outW;
                for (int ox = 0; ox < outW; ox++)
                {
                    int ix0 = ox * strideW - padL;
                    int xLo = Math.Max(0, ix0), xHi = Math.Min(W, ix0 + kW);
                    float best = float.NegativeInfinity;
                    for (int iy = yLo; iy < yHi; iy++)
                    {
                        int row = xBase + iy * W;
                        for (int ix = xLo; ix < xHi; ix++)
                        {
                            float v = xd[row + ix];
                            if (v > best) best = v;
                        }
                    }
                    yd[yRow + ox] = best;
                }
            }
        });

        return y;
    }

    /// <summary>
    /// 2-D average pooling over NCHW input (floor output shape — ceil_mode not supported
    /// by this kernel). <paramref name="countIncludePad"/> false divides each window's sum
    /// by the number of non-pad elements it covers; true divides by kH·kW. Empty attribute
    /// arrays mean ONNX defaults: strides [1,1], pads [0,0,0,0] ([top, left, bottom, right]).
    /// </summary>
    public static Tensor AveragePool(Tensor x, long[] kernel, long[] strides, long[] pads, bool countIncludePad)
    {
        var (N, C, H, W, kH, kW, strideH, strideW, padT, padL, padB, padR) = PoolGeometry(x, kernel, strides, pads, "AveragePool");
        int outH = PoolOutDim(H, kH, strideH, padT, padB, ceilMode: false, "AveragePool");
        int outW = PoolOutDim(W, kW, strideW, padL, padR, ceilMode: false, "AveragePool");

        var y = new Tensor([N, C, outH, outW]);
        float[] xd = x.Data, yd = y.Data;
        int inPlane = H * W, outPlane = outH * outW;
        int kernelArea = kH * kW;

        Parallel.For(0, N * C, nc =>
        {
            int xBase = nc * inPlane;
            int yBase = nc * outPlane;
            for (int oy = 0; oy < outH; oy++)
            {
                int iy0 = oy * strideH - padT;
                int yLo = Math.Max(0, iy0), yHi = Math.Min(H, iy0 + kH);
                int yRow = yBase + oy * outW;
                for (int ox = 0; ox < outW; ox++)
                {
                    int ix0 = ox * strideW - padL;
                    int xLo = Math.Max(0, ix0), xHi = Math.Min(W, ix0 + kW);
                    float sum = 0f;
                    for (int iy = yLo; iy < yHi; iy++)
                    {
                        int row = xBase + iy * W;
                        for (int ix = xLo; ix < xHi; ix++)
                            sum += xd[row + ix];
                    }
                    int count = (yHi - yLo) * (xHi - xLo);         // non-pad elements in the window
                    yd[yRow + ox] = count == 0 ? 0f : sum / (countIncludePad ? kernelArea : count);
                }
            }
        });

        return y;
    }

    /// <summary>Mean over all H·W positions per channel: NCHW [N,C,H,W] → [N,C,1,1].</summary>
    public static Tensor GlobalAveragePool(Tensor x)
    {
        if (x.IsInt64)
            throw new NotSupportedException("GlobalAveragePool supports float32 tensors only.");
        if (x.Rank != 4)
            throw new NotSupportedException($"GlobalAveragePool expects NCHW input, got {x.ShapeString}.");

        int N = x.Shape[0], C = x.Shape[1];
        int plane = x.Shape[2] * x.Shape[3];
        var y = new Tensor([N, C, 1, 1]);
        if (plane == 0) return y;

        float[] xd = x.Data, yd = y.Data;
        for (int nc = 0; nc < N * C; nc++)
        {
            int start = nc * plane;
            double sum = 0.0;                                      // double accumulator: planes can be huge
            for (int i = 0; i < plane; i++)
                sum += xd[start + i];
            yd[nc] = (float)(sum / plane);
        }
        return y;
    }

    /// <summary>Validated 2-D pooling geometry shared by MaxPool and AveragePool.</summary>
    private static (int N, int C, int H, int W, int KH, int KW, int SH, int SW, int PT, int PL, int PB, int PR)
        PoolGeometry(Tensor x, long[] kernel, long[] strides, long[] pads, string op)
    {
        if (x.IsInt64)
            throw new NotSupportedException($"{op} supports float32 tensors only.");
        if (x.Rank != 4)
            throw new NotSupportedException($"{op} expects NCHW input, got {x.ShapeString}.");
        if (kernel.Length == 0)
            throw new ArgumentException($"{op} requires the 'kernel_shape' attribute.");
        if (kernel.Length != 2)
            throw new NotSupportedException($"{op} is 2-D only: 'kernel_shape' must have 2 values, got {kernel.Length}.");

        int kH = checked((int)kernel[0]), kW = checked((int)kernel[1]);
        if (kH <= 0 || kW <= 0)
            throw new ArgumentException($"{op} kernel must be positive, got [{kH},{kW}].");

        int strideH = 1, strideW = 1;                              // ONNX default: 1 along each spatial axis
        if (strides.Length != 0)
        {
            if (strides.Length != 2)
                throw new NotSupportedException($"{op} is 2-D only: 'strides' must have 2 values, got {strides.Length}.");
            strideH = checked((int)strides[0]);
            strideW = checked((int)strides[1]);
            if (strideH <= 0 || strideW <= 0)
                throw new ArgumentException($"{op} strides must be positive, got [{strideH},{strideW}].");
        }

        int padT = 0, padL = 0, padB = 0, padR = 0;
        if (pads.Length != 0)
        {
            if (pads.Length != 4)
                throw new NotSupportedException($"{op} is 2-D only: 'pads' must have 4 values [top,left,bottom,right], got {pads.Length}.");
            padT = checked((int)pads[0]);
            padL = checked((int)pads[1]);
            padB = checked((int)pads[2]);
            padR = checked((int)pads[3]);
            if (padT < 0 || padL < 0 || padB < 0 || padR < 0)
                throw new ArgumentException($"{op} pads must be non-negative, got [{padT},{padL},{padB},{padR}].");
        }

        return (x.Shape[0], x.Shape[1], x.Shape[2], x.Shape[3], kH, kW, strideH, strideW, padT, padL, padB, padR);
    }

    /// <summary>
    /// Output extent along one spatial axis: floor or ceil of (size + pads − k)/stride + 1.
    /// With ceil_mode, ONNX requires the last window to start strictly inside the padded
    /// input; a window starting past its end is dropped (ceil overshoots by at most one).
    /// </summary>
    private static int PoolOutDim(int size, int k, int stride, int padLo, int padHi, bool ceilMode, string op)
    {
        int span = size + padLo + padHi - k;
        if (span < 0)
            throw new ArgumentException($"{op} kernel {k} exceeds padded input extent {size + padLo + padHi}.");
        int outDim = ceilMode ? (span + stride - 1) / stride + 1 : span / stride + 1;
        if (ceilMode && (outDim - 1) * stride >= size + padLo + padHi)
            outDim--;
        return outDim;
    }
}
