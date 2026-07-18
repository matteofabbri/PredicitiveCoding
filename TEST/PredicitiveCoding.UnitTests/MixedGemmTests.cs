using PredictiveCoding.Device;
using PredictiveCoding.LLM.Infrastructure;

namespace PredicitiveCoding.UnitTests;

// Phase 5 — mixed-precision GEMM (16-bit storage, 32-bit accumulation).
// On a CUDA machine CudaGemmProvider runs cublasGemmEx on the tensor cores; in a
// GPU-less sandbox it exercises the CPU mixed fallback. Both must track float GEMM.
public class MixedGemmTests
{
    private static (float[] a, float[] b) RandomMatrices(int m, int k, int n, int seed)
    {
        var rng = new Random(seed);
        var a = new float[m * k];
        var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);
        return (a, b);
    }

    private static void AssertClose(float[] expected, float[] actual)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            float tol = 1e-2f * (1f + MathF.Abs(expected[i]));   // Half rounding over k terms
            Assert.True(MathF.Abs(expected[i] - actual[i]) <= tol,
                $"i={i}: float={expected[i]}, mixed={actual[i]}");
        }
    }

    [Fact]
    public void CpuMixedGemm_MatchesFloatGemm_WithinTolerance()
    {
        const int m = 8, k = 16, n = 4;
        var (a, b) = RandomMatrices(m, k, n, seed: 1);
        var provider = new CpuGemmProvider();

        AssertClose(provider.MatMul(a, m, k, b, n), provider.MatMulMixed(a, m, k, b, n));
    }

    [Fact]
    public void CudaMixedGemm_MatchesFloatGemm_WithinTolerance()
    {
        const int m = 8, k = 16, n = 4;
        var (a, b) = RandomMatrices(m, k, n, seed: 2);

        var cFloat = new CpuGemmProvider().MatMul(a, m, k, b, n);
        var cMixed = new CudaGemmProvider().MatMulMixed(a, m, k, b, n);   // GPU tensor cores, or CPU fallback

        AssertClose(cFloat, cMixed);
    }
}
