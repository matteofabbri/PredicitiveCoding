using System.Runtime.InteropServices;

namespace PredictiveCoding.Device.Interop;

// ─────────────────────────────────────────────────────────────────────────────
// Nvrtc  —  NVIDIA's runtime compiler: turns CUDA C kernel SOURCE into PTX at
// runtime, so custom kernels ship as strings, not as a compiled native DLL.
// This is what lets the non-GEMM ops (GELU, softmax, norm, …) run on the GPU
// while honouring the in-house rule "P/Invoke system libraries only, nothing to
// compile or ship". The logical name "nvrtc" is mapped by NativeResolver.
// ─────────────────────────────────────────────────────────────────────────────

internal static class Nvrtc
{
    [DllImport("nvrtc", EntryPoint = "nvrtcCreateProgram")]
    internal static extern int nvrtcCreateProgram(out IntPtr prog,
        [MarshalAs(UnmanagedType.LPStr)] string src, [MarshalAs(UnmanagedType.LPStr)] string name,
        int numHeaders, IntPtr headers, IntPtr includeNames);

    [DllImport("nvrtc", EntryPoint = "nvrtcCompileProgram")]
    internal static extern int nvrtcCompileProgram(IntPtr prog, int numOptions, IntPtr options);

    [DllImport("nvrtc", EntryPoint = "nvrtcGetPTXSize")]
    internal static extern int nvrtcGetPTXSize(IntPtr prog, out nuint size);

    [DllImport("nvrtc", EntryPoint = "nvrtcGetPTX")]
    internal static extern int nvrtcGetPTX(IntPtr prog, byte[] ptx);

    [DllImport("nvrtc", EntryPoint = "nvrtcGetProgramLogSize")]
    internal static extern int nvrtcGetProgramLogSize(IntPtr prog, out nuint size);

    [DllImport("nvrtc", EntryPoint = "nvrtcGetProgramLog")]
    internal static extern int nvrtcGetProgramLog(IntPtr prog, byte[] log);

    [DllImport("nvrtc", EntryPoint = "nvrtcDestroyProgram")]
    internal static extern int nvrtcDestroyProgram(ref IntPtr prog);
}
