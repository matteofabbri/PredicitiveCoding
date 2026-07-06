namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// Resize — nearest-neighbour up/down-sampling on NCHW tensors.
//
// Covers the configurations real detection nets emit (YOLOX FPN upsample:
// mode=nearest, coordinate_transformation_mode=asymmetric, nearest_mode=floor)
// plus the common half_pixel variant; anything else fails loudly rather than
// silently producing near-correct pixels.
// ─────────────────────────────────────────────────────────────────────────────

public static partial class Ops
{
    /// <summary>ONNX Resize (opset 10-13). Output size comes from <paramref name="sizes"/>
    /// when given, otherwise from <paramref name="scales"/>. 4-D NCHW, nearest mode only;
    /// N and C must keep scale 1.</summary>
    public static Tensor Resize(Tensor x, Tensor? scales, Tensor? sizes,
                                string mode, string coordMode, string nearestMode)
    {
        if (x.IsInt64) throw new NotSupportedException("Resize on int64 tensors is not supported.");
        if (x.Rank != 4) throw new NotSupportedException($"Resize supports 4-D NCHW input, got {x.ShapeString}.");
        if (mode != "nearest")
            throw new NotSupportedException($"Resize mode '{mode}' is not supported (nearest only).");
        if (coordMode is not ("asymmetric" or "half_pixel"))
            throw new NotSupportedException($"Resize coordinate_transformation_mode '{coordMode}' is not supported.");
        if (nearestMode is not ("floor" or "round_prefer_floor"))
            throw new NotSupportedException($"Resize nearest_mode '{nearestMode}' is not supported.");

        int n = x.Shape[0], c = x.Shape[1], h = x.Shape[2], w = x.Shape[3];

        // Resolve output shape and effective per-axis scale (out/in, as ONNX defines it).
        int outH, outW;
        double scaleH, scaleW;
        if (sizes is { Count: > 0 })
        {
            long[] s = sizes.I64 ?? throw new NotSupportedException("Resize 'sizes' must be an int64 tensor.");
            if (s.Length != 4 || s[0] != n || s[1] != c)
                throw new NotSupportedException("Resize may not change the N or C dimension.");
            outH = checked((int)s[2]);
            outW = checked((int)s[3]);
            scaleH = (double)outH / h;
            scaleW = (double)outW / w;
        }
        else if (scales is { Count: > 0 })
        {
            float[] s = scales.Data.Length == 4 ? scales.Data
                : throw new NotSupportedException($"Resize 'scales' must hold 4 floats, got {scales.Count}.");
            if (Math.Abs(s[0] - 1f) > 1e-6 || Math.Abs(s[1] - 1f) > 1e-6)
                throw new NotSupportedException("Resize may not change the N or C dimension.");
            scaleH = s[2];
            scaleW = s[3];
            outH = (int)Math.Floor(h * scaleH);
            outW = (int)Math.Floor(w * scaleW);
        }
        else
        {
            throw new NotSupportedException("Resize needs a non-empty 'scales' or 'sizes' input.");
        }

        if (outH <= 0 || outW <= 0)
            throw new ArgumentException($"Resize computed empty output {outH}×{outW}.");

        // Precompute the source index for every output row/column once.
        int[] srcY = MapNearest(outH, h, scaleH, coordMode, nearestMode);
        int[] srcX = MapNearest(outW, w, scaleW, coordMode, nearestMode);

        var y = new Tensor([n, c, outH, outW]);
        int inPlane = h * w, outPlane = outH * outW;
        System.Threading.Tasks.Parallel.For(0, n * c, p =>
        {
            int src = p * inPlane, dst = p * outPlane;
            for (int oy = 0; oy < outH; oy++)
            {
                int rowBase = src + srcY[oy] * w;
                int outRow = dst + oy * outW;
                for (int ox = 0; ox < outW; ox++)
                    y.Data[outRow + ox] = x.Data[rowBase + srcX[ox]];
            }
        });
        return y;
    }

    /// <summary>Source index per output index for one axis, per the ONNX coordinate rules.</summary>
    private static int[] MapNearest(int outDim, int inDim, double scale, string coordMode, string nearestMode)
    {
        var map = new int[outDim];
        for (int i = 0; i < outDim; i++)
        {
            double orig = coordMode == "asymmetric"
                ? i / scale
                : (i + 0.5) / scale - 0.5;                       // half_pixel
            double idx = nearestMode == "floor"
                ? Math.Floor(orig)
                : Math.Ceiling(orig - 0.5);                      // round_prefer_floor
            map[i] = (int)Math.Clamp(idx, 0, inDim - 1);
        }
        return map;
    }
}
