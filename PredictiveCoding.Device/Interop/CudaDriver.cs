using System.Runtime.InteropServices;

namespace PredictiveCoding.Device.Interop;

// ─────────────────────────────────────────────────────────────────────────────
// CudaDriver  —  the slice of the CUDA Driver API needed to load a runtime-
// compiled module and launch its kernels: context, module, launch, sync.
//
// The driver API co-exists with the runtime API (cudart) used elsewhere: both
// share the device's PRIMARY context, so kernels launched here run on the same
// context and can read/write pointers allocated by the cudart allocator. The
// logical name "cuda" is mapped to nvcuda.dll / libcuda.so by NativeResolver.
// ─────────────────────────────────────────────────────────────────────────────

internal static class CudaDriver
{
    [DllImport("cuda", EntryPoint = "cuInit")]
    internal static extern int cuInit(uint flags);

    [DllImport("cuda", EntryPoint = "cuDeviceGet")]
    internal static extern int cuDeviceGet(out int device, int ordinal);

    [DllImport("cuda", EntryPoint = "cuDevicePrimaryCtxRetain")]
    internal static extern int cuDevicePrimaryCtxRetain(out IntPtr ctx, int device);

    [DllImport("cuda", EntryPoint = "cuCtxSetCurrent")]
    internal static extern int cuCtxSetCurrent(IntPtr ctx);

    [DllImport("cuda", EntryPoint = "cuModuleLoadData")]
    internal static extern int cuModuleLoadData(out IntPtr module, byte[] image);

    [DllImport("cuda", EntryPoint = "cuModuleGetFunction")]
    internal static extern int cuModuleGetFunction(out IntPtr func, IntPtr module, [MarshalAs(UnmanagedType.LPStr)] string name);

    [DllImport("cuda", EntryPoint = "cuLaunchKernel")]
    internal static extern int cuLaunchKernel(
        IntPtr func,
        uint gridX, uint gridY, uint gridZ,
        uint blockX, uint blockY, uint blockZ,
        uint sharedMemBytes, IntPtr stream, IntPtr kernelParams, IntPtr extra);

    [DllImport("cuda", EntryPoint = "cuCtxSynchronize")]
    internal static extern int cuCtxSynchronize();
}
