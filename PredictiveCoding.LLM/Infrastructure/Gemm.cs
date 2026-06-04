using System.Numerics;
using PredictiveCoding.Core;

namespace PredictiveCoding.LLM.Infrastructure;

// ─────────────────────────────────────────────────────────────────────────────
// Gemm  —  general matrix multiply (GEMM) with a pluggable backend.
//
// The default CpuGemmProvider is multithreaded (Parallel.For) and SIMD-vectorised
// (System.Numerics.Vector) using a cache-friendly i-k-j loop. A GPU backend
// (CUDA / Metal) would implement IGemmProvider and be assigned to Gemm.Provider;
// the rest of the model is backend-agnostic.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Pluggable GEMM backend. C[m,n] = A[m,k] · B[k,n] (row-major).</summary>
public interface IGemmProvider
{
    float[] MatMul(float[] a, int m, int k, float[] b, int n);
}

/// <summary>Multithreaded + SIMD CPU GEMM.</summary>
public sealed class CpuGemmProvider : IGemmProvider
{
    public float[] MatMul(float[] a, int m, int k, float[] b, int n)
    {
        var c = new float[m * n];
        Parallel.For(0, m, i => Gemm.RowIkj(a, i, k, b, n, c));
        return c;
    }
}

public static class Gemm
{
    /// <summary>The active GEMM backend (default: CPU). Swap for a GPU provider.</summary>
    public static IGemmProvider Provider { get; set; } = new CpuGemmProvider();

    public static float[] MatMul(float[] a, int m, int k, float[] b, int n)
    {
        if (a.Length != m * k) throw new ArgumentException("A has wrong length.", nameof(a));
        if (b.Length != k * n) throw new ArgumentException("B has wrong length.", nameof(b));
        return Provider.MatMul(a, m, k, b, n);
    }

    /// <summary>
    /// Batched GEMM: multiplies <paramref name="batch"/> independent (A,B) pairs in
    /// parallel over the batch (each product computed sequentially to avoid
    /// thread oversubscription).
    /// </summary>
    public static float[][] BatchedMatMul((float[] a, float[] b)[] batch, int m, int k, int n)
    {
        var outputs = new float[batch.Length][];
        Parallel.For(0, batch.Length, idx =>
        {
            var c = new float[m * n];
            for (int i = 0; i < m; i++) RowIkj(batch[idx].a, i, k, batch[idx].b, n, c);
            outputs[idx] = c;
        });
        return outputs;
    }

    /// <summary>y = W·x for W[out,in], SIMD dot product parallel over output rows.</summary>
    public static DenseTensor<float> MatVec(DenseTensor<float> w, DenseTensor<float> x)
    {
        int rows = w.Dimensions[0], cols = w.Dimensions[1];
        if (x.Length != cols) throw new ArgumentException($"x has {x.Length} elements; expected {cols}.");
        var y  = new DenseTensor<float>(rows);
        var wb = w.Buffer; var xb = x.Buffer; var yb = y.Buffer;
        Parallel.For(0, rows, r => yb[r] = Dot(wb.AsSpan(r * cols, cols), xb.AsSpan(0, cols)));
        return y;
    }

    // ── kernels ──────────────────────────────────────────────────────────────

    // Accumulates one output row i: C[i,:] += Σ_l A[i,l]·B[l,:]  (SIMD over j).
    internal static void RowIkj(float[] a, int i, int k, float[] b, int n, float[] c)
    {
        var crow = c.AsSpan(i * n, n);
        int width = Vector<float>.Count;
        for (int l = 0; l < k; l++)
        {
            float av = a[i * k + l];
            if (av == 0f) continue;
            var brow = b.AsSpan(l * n, n);
            var vav  = new Vector<float>(av);
            int j = 0;
            for (; j <= n - width; j += width)
            {
                var acc = new Vector<float>(crow.Slice(j)) + vav * new Vector<float>(brow.Slice(j));
                acc.CopyTo(crow.Slice(j));
            }
            for (; j < n; j++) crow[j] += av * brow[j];
        }
    }

    internal static float Dot(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        int n = a.Length, width = Vector<float>.Count, i = 0;
        var acc = Vector<float>.Zero;
        for (; i <= n - width; i += width)
            acc += new Vector<float>(a.Slice(i)) * new Vector<float>(b.Slice(i));
        float s = Vector.Sum(acc);
        for (; i < n; i++) s += a[i] * b[i];
        return s;
    }
}
