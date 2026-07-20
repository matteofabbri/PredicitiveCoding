using System.Runtime.InteropServices;

namespace PredictiveCoding.Device.Interop;

// ─────────────────────────────────────────────────────────────────────────────
// Hip / Hipblas  —  P/Invoke bindings to the ROCm runtime (amdhip64) and
// hipBLAS, mirroring Cuda.cs / Cublas.cs exactly (ROADMAP-Scale A4).
//
// ⚠ UNTESTED-WITHOUT-HARDWARE: no AMD GPU exists on any project machine (the
// server's iGPU is not a ROCm compute device, the dev PC is NVIDIA). These
// bindings mirror the verified CUDA path 1:1 and load lazily — with no ROCm
// runtime present every probe throws DllNotFound and the stack stays on CPU —
// but the hip code path itself has never executed on silicon. Treat it as
// portability plumbing, not as a verified backend, until a supported Radeon
// (or a Linux ROCm box) exists.
//
// API notes vs CUDA: hipMemcpyKind uses the same 0..4 ordering as cudaMemcpyKind;
// hipblasOperation_t does NOT match cublasOperation_t — N=111, T=112, C=113.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>hipBLAS transpose operation (hipblasOperation_t: N=111, T=112, C=113 — not cuBLAS's 0/1/2).</summary>
public enum HipblasOp { N = 111, T = 112, C = 113 }

internal static class Hip
{
    private const string Lib = "amdhip64";
    private const CallingConvention Cdecl = CallingConvention.Cdecl;

    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int hipGetDeviceCount(out int count);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int hipSetDevice(int deviceId);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int hipMalloc(out IntPtr ptr, nuint size);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int hipFree(IntPtr ptr);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int hipMemcpy(IntPtr dst, IntPtr src, nuint count, int kind);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int hipDeviceSynchronize();
}

internal static class Hipblas
{
    private const string Lib = "hipblas";
    private const CallingConvention Cdecl = CallingConvention.Cdecl;

    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int hipblasCreate(out IntPtr handle);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int hipblasDestroy(IntPtr handle);

    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)]
    internal static extern int hipblasSgemm(
        IntPtr handle, int transa, int transb, int m, int n, int k,
        ref float alpha, IntPtr a, int lda, IntPtr b, int ldb,
        ref float beta, IntPtr c, int ldc);
}
