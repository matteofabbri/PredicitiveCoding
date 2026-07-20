using System.Reflection;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace PredictiveCoding.Device.Interop;

// ─────────────────────────────────────────────────────────────────────────────
// NativeResolver  —  maps the logical names "cudart" / "cublas" to the real,
// version-suffixed CUDA Toolkit libraries at load time.
//
// We P/Invoke the libraries that ship with the CUDA Toolkit / NVIDIA driver
// directly — there is NO custom native DLL to compile or ship. A module
// initializer registers the resolver before any P/Invoke runs; if the libraries
// are absent, loading fails and the managed layer falls back to the CPU.
// ─────────────────────────────────────────────────────────────────────────────

internal static class NativeResolver
{
    [ModuleInitializer]
    internal static void Register()
        => NativeLibrary.SetDllImportResolver(typeof(NativeResolver).Assembly, Resolve);

    private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        string[] candidates = libraryName switch
        {
            "cudart" => Names("cudart", "cudart64"),
            "cublas" => Names("cublas", "cublas64"),
            // Driver API and the runtime kernel compiler (Phase-A device kernels).
            // The driver lib is unversioned (nvcuda.dll / libcuda.so); NVRTC carries a
            // two-part API-version suffix on Windows (nvrtc64_120_0 across CUDA 12.x).
            "cuda"   => RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                            ? ["nvcuda"]
                            : ["libcuda.so.1", "libcuda.so", "cuda"],
            "nvrtc"  => RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                            ? ["nvrtc64_120_0", "nvrtc64_112_0"]
                            : ["libnvrtc.so.12", "libnvrtc.so.11", "libnvrtc.so"],
            // ROCm mirror (A4, UNTESTED-WITHOUT-HARDWARE): the HIP runtime is
            // version-suffixed on both OSes; hipBLAS ships as a plain name.
            "amdhip64" => RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                            ? ["amdhip64_6", "amdhip64"]
                            : ["libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so"],
            "hipblas"  => RuntimeInformation.IsOSPlatform(OSPlatform.Windows)
                            ? ["hipblas"]
                            : ["libhipblas.so.2", "libhipblas.so.1", "libhipblas.so"],
            _        => [libraryName],
        };

        foreach (var name in candidates)
            if (NativeLibrary.TryLoad(name, out IntPtr handle)) return handle;

        return IntPtr.Zero;   // not found → default handling (caller catches DllNotFound)
    }

    // Common version-suffixed file names across Windows / Linux for a CUDA library.
    private static string[] Names(string posix, string winPrefix)
    {
        if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
            return [$"{winPrefix}_12", $"{winPrefix}_120", $"{winPrefix}_11", $"{winPrefix}_110", posix];
        return [$"lib{posix}.so.12", $"lib{posix}.so.11", $"lib{posix}.so", posix];
    }
}
