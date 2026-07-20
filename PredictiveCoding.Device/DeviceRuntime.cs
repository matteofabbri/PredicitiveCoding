using System.Runtime.InteropServices;
using PredictiveCoding.Device.Interop;

namespace PredictiveCoding.Device;

// ─────────────────────────────────────────────────────────────────────────────
// DeviceRuntime  —  detects the CUDA Toolkit libraries and available GPUs.
//
// GPU work P/Invokes the standard cudart / cublas libraries directly (no custom
// DLL). If those libraries (or a GPU) are absent, cudaGetDeviceCount throws
// DllNotFound, CudaAvailable becomes false, and the whole stack runs on the CPU.
// ─────────────────────────────────────────────────────────────────────────────

public static class DeviceRuntime
{
    private static readonly Lazy<int>  _deviceCount  = new(SafeDeviceCount);
    private static readonly Lazy<bool> _nativeLoaded = new(SafeNativeLoaded);

    /// <summary>
    /// True when the CUDA runtime library could be loaded. Probes through the
    /// actual P/Invoke (which goes through <see cref="Interop.NativeResolver"/>'s
    /// version-suffixed names) — <c>NativeLibrary.TryLoad("cudart")</c> would miss
    /// a perfectly loadable <c>cudart64_12.dll</c> on Windows.
    /// </summary>
    public static bool NativeLoaded => _nativeLoaded.Value;

    private static bool SafeNativeLoaded()
    {
        try { Cuda.cudaGetDeviceCount(out _); return true; }
        catch (DllNotFoundException) { return false; }
        catch (EntryPointNotFoundException) { return false; }
    }

    /// <summary>Number of CUDA devices (0 when unavailable).</summary>
    public static int CudaDeviceCount => _deviceCount.Value;

    /// <summary>True when at least one CUDA device is usable.</summary>
    public static bool CudaAvailable => CudaDeviceCount > 0;

    /// <summary>True when more than one CUDA device is present (multi-GPU).</summary>
    public static bool MultiGpuAvailable => CudaDeviceCount > 1;

    private static readonly Lazy<int> _rocmDeviceCount = new(SafeRocmDeviceCount);

    /// <summary>
    /// Number of ROCm (HIP) devices — 0 when the runtime or a GPU is absent.
    /// A4 mirror, UNTESTED-WITHOUT-HARDWARE: probing works everywhere (a
    /// missing amdhip64 throws DllNotFound → 0), the positive path has never
    /// seen an AMD GPU.
    /// </summary>
    public static int RocmDeviceCount => _rocmDeviceCount.Value;

    /// <summary>True when at least one ROCm device is usable.</summary>
    public static bool RocmAvailable => RocmDeviceCount > 0;

    /// <summary>Resolves a requested device to one that actually exists (CUDA → CPU fallback).</summary>
    public static Device Resolve(Device requested)
    {
        if (requested.IsCuda && !CudaAvailable) return Device.Cpu;
        if (requested.Kind == DeviceKind.CudaMulti && !MultiGpuAvailable)
            return CudaAvailable ? Device.Cuda(0) : Device.Cpu;
        // ROCm has a GEMM provider (A4) but no tensor-placement backend yet;
        // Metal has neither. Tensor placement falls back to the CPU for both.
        if (requested.Kind is DeviceKind.Metal or DeviceKind.Rocm) return Device.Cpu;
        return requested;
    }

    private static int SafeDeviceCount()
    {
        try { return Cuda.cudaGetDeviceCount(out int c) == 0 ? c : 0; }
        catch (DllNotFoundException) { return 0; }
        catch (EntryPointNotFoundException) { return 0; }
        catch { return 0; }
    }

    private static int SafeRocmDeviceCount()
    {
        try { return Hip.hipGetDeviceCount(out int c) == 0 ? c : 0; }
        catch (DllNotFoundException) { return 0; }
        catch (EntryPointNotFoundException) { return 0; }
        catch { return 0; }
    }
}
