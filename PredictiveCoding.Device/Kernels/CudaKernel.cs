using System.Runtime.InteropServices;
using System.Text;
using PredictiveCoding.Device.Interop;

namespace PredictiveCoding.Device.Kernels;

// ─────────────────────────────────────────────────────────────────────────────
// CudaKernel  —  a runtime-compiled GPU kernel, ready to launch.
//
// Compiles a CUDA C source string to PTX (NVRTC), loads it (driver API) and
// caches the function handle. Construct once (e.g. behind a Lazy), launch many
// times. The non-GEMM half of device-resident compute lives on top of this:
// GELU, softmax, RMSNorm, RoPE — each a small kernel string, no shipped DLL.
//
// Launch marshals the arguments (blittable values — device pointers as IntPtr,
// ints, floats) into the void** kernelParams array the driver expects.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class CudaKernel
{
    private readonly IntPtr _func;

    public CudaKernel(string source, string entryName, string virtualFileName = "kernel.cu")
    {
        EnsureContext();

        Check(Nvrtc.nvrtcCreateProgram(out IntPtr prog, source, virtualFileName, 0, IntPtr.Zero, IntPtr.Zero), "nvrtcCreateProgram");
        int compile = Nvrtc.nvrtcCompileProgram(prog, 0, IntPtr.Zero);
        if (compile != 0)
        {
            Nvrtc.nvrtcGetProgramLogSize(prog, out nuint logSize);
            var log = new byte[(int)Math.Max(1, logSize)];
            Nvrtc.nvrtcGetProgramLog(prog, log);
            throw new InvalidOperationException(
                $"NVRTC failed to compile kernel '{entryName}': {Encoding.ASCII.GetString(log).TrimEnd('\0')}");
        }

        Check(Nvrtc.nvrtcGetPTXSize(prog, out nuint ptxSize), "nvrtcGetPTXSize");
        var ptx = new byte[(int)ptxSize];
        Check(Nvrtc.nvrtcGetPTX(prog, ptx), "nvrtcGetPTX");
        Nvrtc.nvrtcDestroyProgram(ref prog);

        Check(CudaDriver.cuModuleLoadData(out IntPtr module, ptx), "cuModuleLoadData");
        Check(CudaDriver.cuModuleGetFunction(out _func, module, entryName), "cuModuleGetFunction");
    }

    /// <summary>
    /// Launches a 1-D grid covering <paramref name="elementCount"/> threads (block
    /// size <paramref name="blockSize"/>), then blocks until it completes.
    /// <paramref name="args"/> are the kernel parameters in order — pass device
    /// pointers as <see cref="IntPtr"/>, scalars as <c>int</c>/<c>float</c>.
    /// </summary>
    public void Launch1D(int elementCount, int blockSize, params object[] args)
    {
        uint grid = (uint)((elementCount + blockSize - 1) / blockSize);
        Launch(grid, (uint)blockSize, args);
    }

    /// <summary>Launches with an explicit grid/block (1-D), then blocks until it completes.</summary>
    public void Launch(uint gridX, uint blockX, params object[] args)
    {
        var pins = new GCHandle[args.Length];
        var ptrs = new IntPtr[args.Length];
        for (int i = 0; i < args.Length; i++)
        {
            pins[i] = GCHandle.Alloc(args[i], GCHandleType.Pinned);   // boxed blittable → pinned
            ptrs[i] = pins[i].AddrOfPinnedObject();
        }
        var pinnedPtrs = GCHandle.Alloc(ptrs, GCHandleType.Pinned);
        try
        {
            Check(CudaDriver.cuLaunchKernel(_func, gridX, 1, 1, blockX, 1, 1, 0, IntPtr.Zero,
                                            pinnedPtrs.AddrOfPinnedObject(), IntPtr.Zero), "cuLaunchKernel");
            Check(CudaDriver.cuCtxSynchronize(), "cuCtxSynchronize");
        }
        finally
        {
            pinnedPtrs.Free();
            foreach (var h in pins) h.Free();
        }
    }

    // The driver-API module loader needs a current context on the thread. cudart
    // and the driver share the device's primary context, so retaining + setting it
    // current here interoperates with the runtime-API allocator/cuBLAS used elsewhere.
    private static readonly object CtxLock = new();
    private static bool _ctxReady;

    internal static void EnsureContext()
    {
        if (_ctxReady) return;
        lock (CtxLock)
        {
            if (_ctxReady) return;
            Check(CudaDriver.cuInit(0), "cuInit");
            Check(CudaDriver.cuDeviceGet(out int dev, 0), "cuDeviceGet");
            Check(CudaDriver.cuDevicePrimaryCtxRetain(out IntPtr ctx, dev), "cuDevicePrimaryCtxRetain");
            Check(CudaDriver.cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
            _ctxReady = true;
        }
    }

    private static void Check(int result, string what)
    {
        if (result != 0) throw new InvalidOperationException($"CUDA {what} failed (code {result}).");
    }
}
