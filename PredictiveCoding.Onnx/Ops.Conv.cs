namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// Ops.Conv — 2-D convolution, the executor's hot path (166 convs per YOLOX run).
//
// Direct NCHW convolution parallelised over (batch × output channel) planes.
// All index arithmetic is hoisted: the valid kernel row/column ranges are
// precomputed once per output row/column, so the inner loops touch only real
// input pixels and never test padding. 1×1 convolutions — which YOLOX is full
// of — take a dedicated branch that reduces to a per-pixel dot over channels.
// ─────────────────────────────────────────────────────────────────────────────

public static partial class Ops
{
    /// <summary>
    /// 2-D convolution. <paramref name="x"/> is NCHW [N,C,H,W]; <paramref name="w"/> is
    /// [M, C/group, kH, kW]; <paramref name="bias"/> is [M] or null. Empty attribute
    /// arrays mean ONNX defaults: strides [1,1], pads [0,0,0,0], dilations [1,1].
    /// <paramref name="pads"/> is in ONNX order [top, left, bottom, right].
    /// </summary>
    public static Tensor Conv(Tensor x, Tensor w, Tensor? bias, long[] strides, long[] pads, long[] dilations, long group)
    {
        if (x.IsInt64 || w.IsInt64 || (bias is not null && bias.IsInt64))
            throw new NotSupportedException("Conv supports float32 tensors only.");
        if (x.Rank != 4 || w.Rank != 4)
            throw new NotSupportedException(
                $"Conv is 2-D only: expected NCHW input and [M,C/g,kH,kW] weights, got {x.ShapeString} and {w.ShapeString}.");

        var (strideH, strideW) = ConvPair(strides, "strides");
        var (dilH, dilW) = ConvPair(dilations, "dilations");
        var (padT, padL, padB, padR) = ConvPads(pads);

        int N = x.Shape[0], C = x.Shape[1], H = x.Shape[2], W = x.Shape[3];
        int M = w.Shape[0], kC = w.Shape[1], kH = w.Shape[2], kW = w.Shape[3];

        int g = checked((int)group);
        if (g <= 0 || C % g != 0 || M % g != 0)
            throw new ArgumentException($"Conv group={group} must divide input channels {C} and output channels {M}.");
        if (kC != C / g)
            throw new ArgumentException(
                $"Conv weight shape {w.ShapeString} implies {(long)kC * g} input channels, input {x.ShapeString} has {C}.");

        float[]? bd = bias?.Data;
        if (bd is not null && bd.Length != M)
            throw new ArgumentException($"Conv bias has {bd.Length} values, expected {M}.");

        int outH = (H + padT + padB - dilH * (kH - 1) - 1) / strideH + 1;
        int outW = (W + padL + padR - dilW * (kW - 1) - 1) / strideW + 1;
        if (outH <= 0 || outW <= 0)
            throw new ArgumentException(
                $"Conv output is empty for input {x.ShapeString}, kernel {kH}x{kW}, pads [{padT},{padL},{padB},{padR}], strides [{strideH},{strideW}], dilations [{dilH},{dilW}].");

        // Per-output-row/column geometry, hoisted out of the hot loops:
        //   iy0/ix0 — first input row/col the window touches (may point into padding);
        //   kyLo..kyHi / kxLo..kxHi — the kernel taps whose input coordinate lands
        //   inside the image, so the inner loops never branch on padding.
        var iy0 = new int[outH]; var kyLo = new int[outH]; var kyHi = new int[outH];
        for (int oy = 0; oy < outH; oy++)
        {
            int start = oy * strideH - padT;
            iy0[oy] = start;
            kyLo[oy] = start >= 0 ? 0 : (-start + dilH - 1) / dilH;
            int last = H - 1 - start;                              // largest reachable offset from start
            kyHi[oy] = last < 0 ? 0 : Math.Min(kH, last / dilH + 1);
        }
        var ix0 = new int[outW]; var kxLo = new int[outW]; var kxHi = new int[outW];
        for (int ox = 0; ox < outW; ox++)
        {
            int start = ox * strideW - padL;
            ix0[ox] = start;
            kxLo[ox] = start >= 0 ? 0 : (-start + dilW - 1) / dilW;
            int last = W - 1 - start;
            kxHi[ox] = last < 0 ? 0 : Math.Min(kW, last / dilW + 1);
        }

        var y = new Tensor([N, M, outH, outW]);
        float[] xd = x.Data, wd = w.Data, yd = y.Data;

        int inPlane = H * W;                                       // one input channel
        int inImage = C * inPlane;                                 // one batch element
        int kPlane = kH * kW;                                      // one weight channel
        int kBlock = kC * kPlane;                                  // one output channel's weights
        int outPlane = outH * outW;
        int mPerGroup = M / g;

        bool oneByOne = kH == 1 && kW == 1 && padT == 0 && padL == 0 && padB == 0 && padR == 0;

        Parallel.For(0, N * M, nm =>
        {
            int nIdx = nm / M, mIdx = nm % M;
            int xBase = nIdx * inImage + mIdx / mPerGroup * kC * inPlane; // first input channel of this group
            int wBase = mIdx * kBlock;
            int yBase = nm * outPlane;
            float b = bd is not null ? bd[mIdx] : 0f;

            if (oneByOne)
            {
                // 1×1 kernel, no padding: every output pixel is a plain dot product
                // over the group's input channels at one (possibly strided) location.
                for (int oy = 0; oy < outH; oy++)
                {
                    int xRow = xBase + oy * strideH * W;
                    int yRow = yBase + oy * outW;
                    for (int ox = 0; ox < outW; ox++)
                    {
                        float acc = b;
                        for (int cc = 0, xi = xRow + ox * strideW; cc < kC; cc++, xi += inPlane)
                            acc += wd[wBase + cc] * xd[xi];
                        yd[yRow + ox] = acc;
                    }
                }
                return;
            }

            for (int oy = 0; oy < outH; oy++)
            {
                int rowStart = iy0[oy];
                int ky0 = kyLo[oy], ky1 = kyHi[oy];
                int yRow = yBase + oy * outW;
                for (int ox = 0; ox < outW; ox++)
                {
                    int colStart = ix0[ox];
                    int kx0 = kxLo[ox], kx1 = kxHi[ox];
                    float acc = b;
                    int xChan = xBase, wChan = wBase;
                    for (int cc = 0; cc < kC; cc++, xChan += inPlane, wChan += kPlane)
                    {
                        for (int ky = ky0; ky < ky1; ky++)
                        {
                            int xRow = xChan + (rowStart + ky * dilH) * W + colStart;
                            int wRow = wChan + ky * kW;
                            for (int kx = kx0, xi = xRow + kx0 * dilW; kx < kx1; kx++, xi += dilW)
                                acc += wd[wRow + kx] * xd[xi];
                        }
                    }
                    yd[yRow + ox] = acc;
                }
            }
        });

        return y;
    }

    /// <summary>Spatial attribute pair with ONNX default [1,1]; 2-D only.</summary>
    private static (int, int) ConvPair(long[] values, string name)
    {
        if (values.Length == 0) return (1, 1);
        if (values.Length != 2)
            throw new NotSupportedException($"Conv is 2-D only: '{name}' must have 2 values, got {values.Length}.");
        int a = checked((int)values[0]), b = checked((int)values[1]);
        if (a <= 0 || b <= 0)
            throw new ArgumentException($"Conv '{name}' values must be positive, got [{a},{b}].");
        return (a, b);
    }

    /// <summary>Pads in ONNX order [top, left, bottom, right] with default all-zero; 2-D only.</summary>
    private static (int, int, int, int) ConvPads(long[] pads)
    {
        if (pads.Length == 0) return (0, 0, 0, 0);
        if (pads.Length != 4)
            throw new NotSupportedException($"Conv is 2-D only: 'pads' must have 4 values [top,left,bottom,right], got {pads.Length}.");
        int t = checked((int)pads[0]), l = checked((int)pads[1]), b = checked((int)pads[2]), r = checked((int)pads[3]);
        if (t < 0 || l < 0 || b < 0 || r < 0)
            throw new ArgumentException($"Conv pads must be non-negative, got [{t},{l},{b},{r}].");
        return (t, l, b, r);
    }
}
