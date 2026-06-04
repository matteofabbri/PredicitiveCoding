using PredictiveCoding.Core;
using PredictiveCoding.LLM.Infrastructure;

namespace PredicitiveCoding.UnitTests;

public class LlmInfrastructureTests
{
    private static DenseTensor<float> Randn(int rows, int cols, Random rng)
    {
        var t = new DenseTensor<float>(rows, cols);
        for (int i = 0; i < t.Length; i++) t.Buffer[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static DenseTensor<float> Randn(int n, Random rng)
    {
        var t = new DenseTensor<float>(n);
        for (int i = 0; i < n; i++) t.Buffer[i] = (float)(rng.NextDouble() * 2 - 1);
        return t;
    }

    private static float RelL2(DenseTensor<float> a, DenseTensor<float> b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a.Buffer[i] - b.Buffer[i];
            num += d * d; den += (double)b.Buffer[i] * b.Buffer[i];
        }
        return (float)Math.Sqrt(num / (den + 1e-12));
    }

    // ── Mixed precision (fp16 / bf16) ──────────────────────────────────────────

    [Theory]
    [InlineData(LowPrecision.Fp16, 1e-2f)]
    [InlineData(LowPrecision.Bf16, 3e-2f)]
    public void LowPrecisionMatVec_ApproximatesFloat_AndHalvesMemory(LowPrecision prec, float tol)
    {
        var rng = new Random(1);
        var w = Randn(16, 24, rng);
        var x = Randn(24, rng);

        var reference = TensorUtils.MatVecMul(w, x);
        var lp        = LowPrecisionMatrix.FromFloat(w, prec);

        Assert.True(RelL2(lp.MatVec(x), reference) < tol);
        Assert.Equal(16L * 24L * 2L, lp.MemoryBytes);       // 16-bit = half of fp32
    }

    [Fact]
    public void BFloat16_RoundTripsExactFloats()
    {
        foreach (float f in new[] { 0f, 1f, -1f, 2f, 0.5f, -0.25f })
            Assert.Equal(f, BFloat16.FromFloat(f).ToFloat(), precision: 3);
    }

    // ── GEMM ───────────────────────────────────────────────────────────────────

    [Fact]
    public void Gemm_MatMul_MatchesNaive()
    {
        var rng = new Random(2);
        int m = 4, k = 5, n = 6;
        var a = Randn(m, k, rng).Buffer;
        var b = Randn(k, n, rng).Buffer;

        var c = Gemm.MatMul(a, m, k, b, n);
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float expected = 0f;
                for (int l = 0; l < k; l++) expected += a[i * k + l] * b[l * n + j];
                Assert.Equal(expected, c[i * n + j], precision: 4);
            }
    }

    [Fact]
    public void Gemm_Batched_MatchesPerItem()
    {
        var rng = new Random(3);
        int m = 3, k = 4, n = 2;
        var a0 = Randn(m, k, rng).Buffer; var b0 = Randn(k, n, rng).Buffer;
        var a1 = Randn(m, k, rng).Buffer; var b1 = Randn(k, n, rng).Buffer;

        var batch = Gemm.BatchedMatMul([(a0, b0), (a1, b1)], m, k, n);
        Assert.Equal(Gemm.MatMul(a0, m, k, b0, n), batch[0]);
        Assert.Equal(Gemm.MatMul(a1, m, k, b1, n), batch[1]);
    }

    [Fact]
    public void Gemm_MatVec_MatchesTensorUtils()
    {
        var rng = new Random(4);
        var w = Randn(20, 17, rng);
        var x = Randn(17, rng);
        Assert.True(RelL2(Gemm.MatVec(w, x), TensorUtils.MatVecMul(w, x)) < 1e-5f);
    }

    // ── Tensor / multi-device parallelism ──────────────────────────────────────

    [Fact]
    public void ShardedLinear_ColumnAndRowParallel_MatchSingleDevice()
    {
        var rng = new Random(5);
        var w = Randn(10, 12, rng);
        var b = Randn(10, rng).Buffer;
        var x = Randn(12, rng);

        // Reference: W·x + b.
        var reference = TensorUtils.MatVecMul(w, x);
        for (int r = 0; r < reference.Length; r++) reference.Buffer[r] += b[r];

        var sharded = new ShardedLinear(w, b, new DeviceMesh(3));
        Assert.True(RelL2(sharded.ForwardColumnParallel(x), reference) < 1e-5f);
        Assert.True(RelL2(sharded.ForwardRowParallel(x),    reference) < 1e-5f);
    }

    [Fact]
    public void AllReduceSum_SumsPartials()
    {
        var reduced = DeviceMesh.AllReduceSum([[1f, 2f, 3f], [10f, 20f, 30f]]);
        Assert.Equal(new[] { 11f, 22f, 33f }, reduced);
    }

    // ── Quantization (int8 / int4) ──────────────────────────────────────────────

    [Fact]
    public void Int8_ApproximatesFloat_AndShrinks4x()
    {
        var rng = new Random(6);
        var w = Randn(16, 32, rng);
        var x = Randn(32, rng);

        var q = Int8Matrix.FromFloat(w);
        Assert.True(RelL2(q.MatVec(x), TensorUtils.MatVecMul(w, x)) < 5e-2f);
        Assert.True(q.MemoryBytes < 16L * 32L * sizeof(float) / 3);  // ≈ 4× smaller
    }

    [Fact]
    public void Int4_ApproximatesFloat_AndShrinks()
    {
        var rng = new Random(7);
        var w = Randn(16, 32, rng);
        var x = Randn(32, rng);

        var q = Int4Matrix.FromFloat(w);
        Assert.True(RelL2(q.MatVec(x), TensorUtils.MatVecMul(w, x)) < 0.35f);
        Assert.True(q.MemoryBytes < 16L * 32L * sizeof(float) / 5);  // ≈ 8× smaller (+ scales)
    }
}
