namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// Gather — index selection along one axis (opset 1/11/13).
//
// In detection nets this almost always appears in shape pipelines:
// Shape → Gather(scalar index) → Unsqueeze → Concat → Reshape, so indices are
// int64 and often rank-0. Output shape is data.shape[:axis] ++ indices.shape ++
// data.shape[axis+1:] — a rank-0 index therefore REMOVES the axis, which is why
// scalars must not be widened to [1] upstream.
// ─────────────────────────────────────────────────────────────────────────────

public static partial class Ops
{
    /// <summary>ONNX Gather: for every element of <paramref name="indices"/> (int64,
    /// any rank, negative values count from the end of the axis) selects the matching
    /// slice of <paramref name="data"/> along <paramref name="axis"/>.</summary>
    public static Tensor Gather(Tensor data, Tensor indices, int axis)
    {
        int rank = data.Rank;
        if (rank == 0)
            throw new ArgumentException("Gather requires data of rank >= 1.");
        if (axis < -rank || axis >= rank)
            throw new ArgumentException($"Gather axis {axis} out of range for rank {rank}.");
        if (axis < 0) axis += rank;

        long[] idx = indices.I64
            ?? throw new ArgumentException($"Gather 'indices' must be an int64 tensor, got float {indices.ShapeString}.");

        // out shape = data.shape[:axis] ++ indices.shape ++ data.shape[axis+1:]
        var outShape = new int[rank - 1 + indices.Rank];
        int w = 0;
        for (int i = 0; i < axis; i++) outShape[w++] = data.Shape[i];
        for (int i = 0; i < indices.Rank; i++) outShape[w++] = indices.Shape[i];
        for (int i = axis + 1; i < rank; i++) outShape[w++] = data.Shape[i];

        int axisDim = data.Shape[axis];
        int outer = 1;
        for (int i = 0; i < axis; i++) outer *= data.Shape[i];
        int inner = 1;
        for (int i = axis + 1; i < rank; i++) inner *= data.Shape[i];

        var result = NewLike(data, outShape);
        Array src = PayloadArray(data), dst = PayloadArray(result);

        for (int o = 0; o < outer; o++)
        {
            for (int j = 0; j < idx.Length; j++)
            {
                long v = idx[j];
                if (v < 0) v += axisDim;
                if (v < 0 || v >= axisDim)
                    throw new ArgumentException($"Gather index {idx[j]} out of range for axis of size {axisDim}.");
                Array.Copy(src, (o * axisDim + (int)v) * inner, dst, ((long)o * idx.Length + j) * inner, inner);
            }
        }
        return result;
    }
}
