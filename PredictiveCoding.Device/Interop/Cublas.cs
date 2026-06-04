using System.Runtime.InteropServices;

namespace PredictiveCoding.Device.Interop;

/// <summary>cuBLAS transpose operation (cublasOperation_t).</summary>
public enum CublasOp { N = 0, T = 1, C = 2 }

// ─────────────────────────────────────────────────────────────────────────────
// Cublas  —  P/Invoke bindings to cuBLAS (libcublas / cublas64_xx).
//
// Standard CUDA Toolkit library; no custom DLL. Returns cublasStatus_t
// (0 = CUBLAS_STATUS_SUCCESS). cuBLAS is column-major, so the managed GEMM passes
// operands swapped to obtain a row-major result.
// ─────────────────────────────────────────────────────────────────────────────

internal static class Cublas
{
    private const string Lib = "cublas";
    private const CallingConvention Cdecl = CallingConvention.Cdecl;

    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cublasCreate_v2(out IntPtr handle);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cublasDestroy_v2(IntPtr handle);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cublasSetStream_v2(IntPtr handle, IntPtr stream);

    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)]
    internal static extern int cublasSgemm_v2(
        IntPtr handle, int transa, int transb, int m, int n, int k,
        ref float alpha, IntPtr a, int lda, IntPtr b, int ldb,
        ref float beta, IntPtr c, int ldc);

    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)]
    internal static extern int cublasSgemmStridedBatched(
        IntPtr handle, int transa, int transb, int m, int n, int k,
        ref float alpha, IntPtr a, int lda, long strideA,
        IntPtr b, int ldb, long strideB,
        ref float beta, IntPtr c, int ldc, long strideC, int batchCount);
}
