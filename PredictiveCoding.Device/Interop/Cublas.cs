using System.Runtime.InteropServices;

namespace PredictiveCoding.Device.Interop;

/// <summary>cuBLAS transpose operation (cublasOperation_t).</summary>
public enum CublasOp { N = 0, T = 1, C = 2 }

/// <summary>Native math data type (cudaDataType_t): R32F=0, R16F (half)=2, R16BF (bfloat16)=14.</summary>
public enum CudaDataType { R32F = 0, R16F = 2, R16BF = 14 }

/// <summary>cuBLAS compute/accumulation type (cublasComputeType_t, CUDA 11+): 16F=64, 32F=68.</summary>
public enum CublasComputeType { Compute16F = 64, Compute32F = 68 }

/// <summary>cuBLAS GEMM algorithm selector (cublasGemmAlgo_t). Default lets cuBLAS choose tensor-op kernels.</summary>
public enum CublasGemmAlgo { Default = -1 }

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

    // Mixed-precision GEMM: A/B in aType/bType (e.g. R16F), C in cType, accumulation
    // in computeType (e.g. Compute32F). alpha/beta are float32 host scalars when
    // computeType is Compute32F. This is the tensor-core path.
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)]
    internal static extern int cublasGemmEx(
        IntPtr handle, int transa, int transb, int m, int n, int k,
        ref float alpha,
        IntPtr a, int aType, int lda,
        IntPtr b, int bType, int ldb,
        ref float beta,
        IntPtr c, int cType, int ldc,
        int computeType, int algo);
}
