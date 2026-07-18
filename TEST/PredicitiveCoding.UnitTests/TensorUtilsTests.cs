using System.Runtime.CompilerServices;
using PredictiveCoding.Core;

namespace PredicitiveCoding.UnitTests;

public class TensorUtilsTests
{
    // ── MatVecMul ─────────────────────────────────────────────────────────────

    [Fact]
    public void MatVecMul_IdentityMatrix_ReturnsVector()
    {
        // W = I (2×2), v = [3, 5]  →  y = [3, 5]
        var w = new DenseTensor<float>([1, 0, 0, 1], 2, 2);
        var v = new DenseTensor<float>([3f, 5f], 2);

        var y = TensorUtils.MatVecMul(w, v);

        Assert.Equal(2, y.Length);
        Assert.Equal(3f, y[0], precision: 5);
        Assert.Equal(5f, y[1], precision: 5);
    }

    [Fact]
    public void MatVecMul_KnownMatrix_CorrectResult()
    {
        // W = [[1,2],[3,4]], v = [1,1]  →  y = [3, 7]
        var w = new DenseTensor<float>([1, 2, 3, 4], 2, 2);
        var v = new DenseTensor<float>([1f, 1f], 2);

        var y = TensorUtils.MatVecMul(w, v);

        Assert.Equal(3f, y[0], precision: 5);
        Assert.Equal(7f, y[1], precision: 5);
    }

    // ── MatVecMulT ────────────────────────────────────────────────────────────

    [Fact]
    public void MatVecMulT_KnownMatrix_CorrectTransposeResult()
    {
        // W = [[1,2],[3,4]], v = [1,1]  →  W^T @ v = [4, 6]
        var w = new DenseTensor<float>([1, 2, 3, 4], 2, 2);
        var v = new DenseTensor<float>([1f, 1f], 2);

        var y = TensorUtils.MatVecMulT(w, v);

        Assert.Equal(4f, y[0], precision: 5);
        Assert.Equal(6f, y[1], precision: 5);
    }

    // ── OuterProduct ──────────────────────────────────────────────────────────

    [Fact]
    public void OuterProduct_TwoVectors_CorrectMatrix()
    {
        // a = [1,2], b = [3,4]  →  C = [[3,4],[6,8]]
        var a = new DenseTensor<float>([1f, 2f], 2);
        var b = new DenseTensor<float>([3f, 4f], 2);

        var c = TensorUtils.OuterProduct(a, b);

        Assert.Equal(2, c.Dimensions[0]);
        Assert.Equal(2, c.Dimensions[1]);
        Assert.Equal(3f, c[0, 0], precision: 5);
        Assert.Equal(4f, c[0, 1], precision: 5);
        Assert.Equal(6f, c[1, 0], precision: 5);
        Assert.Equal(8f, c[1, 1], precision: 5);
    }

    // ── ElementwiseAdd / Sub / Mul ────────────────────────────────────────────

    [Fact]
    public void ElementwiseAdd_ReturnsCorrectSum()
    {
        var x = new DenseTensor<float>([1f, 2f, 3f], 3);
        var y = new DenseTensor<float>([4f, 5f, 6f], 3);

        var z = TensorUtils.ElementwiseAdd(x, y);

        Assert.Equal(5f, z[0], precision: 5);
        Assert.Equal(7f, z[1], precision: 5);
        Assert.Equal(9f, z[2], precision: 5);
    }

    [Fact]
    public void ElementwiseSub_ReturnsCorrectDifference()
    {
        var x = new DenseTensor<float>([10f, 7f, 3f], 3);
        var y = new DenseTensor<float>([1f,  2f, 3f], 3);

        var z = TensorUtils.ElementwiseSub(x, y);

        Assert.Equal(9f, z[0], precision: 5);
        Assert.Equal(5f, z[1], precision: 5);
        Assert.Equal(0f, z[2], precision: 5);
    }

    [Fact]
    public void ElementwiseMul_ReturnsCorrectProduct()
    {
        var x = new DenseTensor<float>([2f, 3f, 4f], 3);
        var y = new DenseTensor<float>([5f, 6f, 7f], 3);

        var z = TensorUtils.ElementwiseMul(x, y);

        Assert.Equal(10f, z[0], precision: 5);
        Assert.Equal(18f, z[1], precision: 5);
        Assert.Equal(28f, z[2], precision: 5);
    }

    // ── InPlaceAxpy ───────────────────────────────────────────────────────────

    [Fact]
    public void InPlaceAxpy_AddsScaledSource()
    {
        var dst = new DenseTensor<float>([1f, 2f, 3f], 3);
        var src = new DenseTensor<float>([1f, 1f, 1f], 3);

        TensorUtils.InPlaceAxpy(2f, src, dst);

        Assert.Equal(3f, dst[0], precision: 5);
        Assert.Equal(4f, dst[1], precision: 5);
        Assert.Equal(5f, dst[2], precision: 5);
    }

    // ── SumSquares ────────────────────────────────────────────────────────────

    [Fact]
    public void SumSquares_CorrectValue()
    {
        var x = new DenseTensor<float>([3f, 4f], 2);

        float result = TensorUtils.SumSquares(x);

        Assert.Equal(25f, result, precision: 5);
    }

    // ── Argmax ────────────────────────────────────────────────────────────────

    [Fact]
    public void Argmax_ReturnsIndexOfMaximum()
    {
        var x = new DenseTensor<float>([0.1f, 0.9f, 0.3f, 0.5f], 4);

        int idx = TensorUtils.Argmax(x);

        Assert.Equal(1, idx);
    }

    // ── Transpose2D ───────────────────────────────────────────────────────────

    [Fact]
    public void Transpose2D_KnownMatrix_CorrectResult()
    {
        // W = [[1,2,3],[4,5,6]]  (2×3)  →  W^T = [[1,4],[2,5],[3,6]]  (3×2)
        var w = new DenseTensor<float>([1, 2, 3, 4, 5, 6], 2, 3);

        var wt = TensorUtils.Transpose2D(w);

        Assert.Equal(3, wt.Dimensions[0]);
        Assert.Equal(2, wt.Dimensions[1]);
        Assert.Equal(1f, wt[0, 0], precision: 5);
        Assert.Equal(4f, wt[0, 1], precision: 5);
        Assert.Equal(2f, wt[1, 0], precision: 5);
        Assert.Equal(5f, wt[1, 1], precision: 5);
        Assert.Equal(3f, wt[2, 0], precision: 5);
        Assert.Equal(6f, wt[2, 1], precision: 5);
    }

    // ── Concat ────────────────────────────────────────────────────────────────

    [Fact]
    public void Concat_TwoVectors_ConcatenatesCorrectly()
    {
        var a = new DenseTensor<float>([1f, 2f], 2);
        var b = new DenseTensor<float>([3f, 4f, 5f], 3);

        var result = TensorUtils.Concat([a, b]);

        Assert.Equal(5, result.Length);
        Assert.Equal(1f, result[0]);
        Assert.Equal(3f, result[2]);
        Assert.Equal(5f, result[4]);
    }

    // ── Clip ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Clip_ClampsValuesWithinRange()
    {
        var x = new DenseTensor<float>([-2f, 0.5f, 3f], 3);

        var z = TensorUtils.Clip(x, 0f, 1f);

        Assert.Equal(0f, z[0], precision: 5);
        Assert.Equal(0.5f, z[1], precision: 5);
        Assert.Equal(1f, z[2], precision: 5);
    }

    // ── MatVecMulMixed (precisione mista: pesi Half, accumulo float) ───────────

    [Fact]
    public void MatVecMulMixed_MatchesFloatWithinTolerance()
    {
        // Stessi pesi in float e in Half; la versione mista deve coincidere con
        // quella float entro la tolleranza di quantizzazione di Half (~0.1%).
        float[] wData =
        [
             1.5f, -2.25f,  0.5f,  3.0f,
            -1.0f,  0.75f,  2.5f, -0.5f,
             0.25f, 1.25f, -3.5f,  2.0f,
        ];
        var wFloat = new DenseTensor<float>(wData, 3, 4);
        var wHalf  = new DenseTensor<Half>(Array.ConvertAll(wData, x => (Half)x), 3, 4);
        var v = new DenseTensor<float>([2f, -1f, 0.5f, 1f], 4);

        var yFloat = TensorUtils.MatVecMul(wFloat, v);
        var yMixed = TensorUtils.MatVecMulMixed(wHalf, v);

        Assert.Equal(yFloat.Length, yMixed.Length);
        for (int i = 0; i < yFloat.Length; i++)
        {
            float tol = 1e-2f * (1f + Math.Abs(yFloat[i]));   // ~1% relativo, robusto vicino a 0
            Assert.True(Math.Abs(yFloat[i] - yMixed[i]) <= tol,
                $"i={i}: float={yFloat[i]}, mixed={yMixed[i]}");
        }
    }

    [Fact]
    public void MatVecMulMixed_HalfWeightsUseHalfTheMemory()
    {
        // Il punto della precisione mista: i pesi Half occupano metà dei byte.
        var wFloat = new DenseTensor<float>(64, 128);
        var wHalf  = new DenseTensor<Half>(64, 128);

        long floatBytes = (long)wFloat.Buffer.Length * sizeof(float);          // 4 byte/elem
        long halfBytes  = (long)wHalf.Buffer.Length  * Unsafe.SizeOf<Half>();  // 2 byte/elem

        Assert.Equal(2, Unsafe.SizeOf<Half>());
        Assert.Equal(floatBytes / 2, halfBytes);
    }

    [Fact]
    public void MatVecMulTMixed_MatchesFloatWithinTolerance()
    {
        // W^T @ v con pesi Half deve coincidere con la versione float entro Half.
        float[] wData =
        [
             1.5f, -2.25f,  0.5f,  3.0f,
            -1.0f,  0.75f,  2.5f, -0.5f,
             0.25f, 1.25f, -3.5f,  2.0f,
        ];
        var wFloat = new DenseTensor<float>(wData, 3, 4);
        var wHalf  = new DenseTensor<Half>(Array.ConvertAll(wData, x => (Half)x), 3, 4);
        var v = new DenseTensor<float>([2f, -1f, 0.5f], 3);   // lunghezza m=3

        var yFloat = TensorUtils.MatVecMulT(wFloat, v);
        var yMixed = TensorUtils.MatVecMulTMixed(wHalf, v);

        Assert.Equal(yFloat.Length, yMixed.Length);
        for (int i = 0; i < yFloat.Length; i++)
        {
            float tol = 1e-2f * (1f + Math.Abs(yFloat[i]));
            Assert.True(Math.Abs(yFloat[i] - yMixed[i]) <= tol,
                $"i={i}: float={yFloat[i]}, mixed={yMixed[i]}");
        }
    }
}
