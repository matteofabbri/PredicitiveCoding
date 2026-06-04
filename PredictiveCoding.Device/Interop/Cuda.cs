using System.Runtime.InteropServices;

namespace PredictiveCoding.Device.Interop;

/// <summary>Direction flag for <see cref="Cuda.cudaMemcpyAsync"/> (matches cudaMemcpyKind).</summary>
public enum MemcpyKind { HostToHost = 0, HostToDevice = 1, DeviceToHost = 2, DeviceToDevice = 3, Default = 4 }

// ─────────────────────────────────────────────────────────────────────────────
// Cuda  —  P/Invoke bindings to the CUDA Runtime API (libcudart / cudart64_xx).
//
// These are the standard libraries from the CUDA Toolkit; no custom DLL is built.
// Every function returns a cudaError_t (0 = cudaSuccess). The managed layer only
// calls them when DeviceRuntime.CudaAvailable is true.
// ─────────────────────────────────────────────────────────────────────────────

internal static class Cuda
{
    private const string Lib = "cudart";
    private const CallingConvention Cdecl = CallingConvention.Cdecl;
    private const uint cudaHostAllocDefault = 0;

    // ── Device / context ───────────────────────────────────────────────────────
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaGetDeviceCount(out int count);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaSetDevice(int device);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaDeviceSynchronize();

    // ── Memory ─────────────────────────────────────────────────────────────────
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaMalloc(out IntPtr ptr, nuint size);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaFree(IntPtr ptr);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaMallocAsync(out IntPtr ptr, nuint size, IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaFreeAsync(IntPtr ptr, IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaHostAlloc(out IntPtr ptr, nuint size, uint flags);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaFreeHost(IntPtr ptr);

    internal static int cudaHostAlloc(out IntPtr ptr, nuint size) => cudaHostAlloc(out ptr, size, cudaHostAllocDefault);

    // ── Copies ─────────────────────────────────────────────────────────────────
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaMemcpy(IntPtr dst, IntPtr src, nuint count, int kind);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaMemcpyAsync(IntPtr dst, IntPtr src, nuint count, int kind, IntPtr stream);

    // ── Streams / events ───────────────────────────────────────────────────────
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaStreamCreate(out IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaStreamDestroy(IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaStreamSynchronize(IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaEventCreate(out IntPtr evt);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaEventRecord(IntPtr evt, IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaEventSynchronize(IntPtr evt);
    [DllImport(Lib, CallingConvention = Cdecl, ExactSpelling = true)] internal static extern int cudaEventDestroy(IntPtr evt);
}
