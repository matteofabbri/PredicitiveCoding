using System.Runtime.InteropServices;
using PredictiveCoding.Device.Interop;

namespace PredictiveCoding.Device;

// ─────────────────────────────────────────────────────────────────────────────
// DeviceRuntime  —  detects the native bridge and the available accelerators.
//
// Everything here degrades gracefully: if "pcdevice" cannot be loaded (CPU-only
// machine, no CUDA toolkit, etc.) CudaAvailable is false and the whole stack runs
// on the CPU. The native library is probed once via NativeLibrary.TryLoad so a
// missing DLL never throws.
// ─────────────────────────────────────────────────────────────────────────────

public static class DeviceRuntime
{
    private const string Lib = "pcdevice";

    private static readonly bool _nativeLoaded;
    private static readonly Lazy<int> _deviceCount = new(SafeDeviceCount);

    static DeviceRuntime()
    {
        _nativeLoaded = NativeLibrary.TryLoad(Lib, typeof(DeviceRuntime).Assembly, null, out _);
        if (_nativeLoaded)
        {
            try { NativeMethods.pcd_init(); } catch { _nativeLoaded = false; }
        }
    }

    /// <summary>True when the native bridge loaded successfully.</summary>
    public static bool NativeLoaded => _nativeLoaded;

    /// <summary>Number of CUDA devices (0 when unavailable).</summary>
    public static int CudaDeviceCount => _nativeLoaded ? _deviceCount.Value : 0;

    /// <summary>True when at least one CUDA device is usable.</summary>
    public static bool CudaAvailable => CudaDeviceCount > 0;

    /// <summary>True when more than one CUDA device is present (multi-GPU collectives).</summary>
    public static bool MultiGpuAvailable => CudaDeviceCount > 1;

    /// <summary>Resolves a requested device to one that actually exists (CUDA → CPU fallback).</summary>
    public static Device Resolve(Device requested)
    {
        if (requested.IsCuda && !CudaAvailable) return Device.Cpu;
        if (requested.Kind == DeviceKind.CudaMulti && !MultiGpuAvailable)
            return CudaAvailable ? Device.Cuda(0) : Device.Cpu;       // multi → single → cpu
        if (requested.Kind is DeviceKind.Metal or DeviceKind.Rocm) return Device.Cpu; // not built yet
        return requested;
    }

    private static int SafeDeviceCount()
    {
        try { return NativeMethods.pcd_cuda_device_count(); }
        catch (DllNotFoundException) { return 0; }
        catch (EntryPointNotFoundException) { return 0; }
        catch { return 0; }
    }
}
