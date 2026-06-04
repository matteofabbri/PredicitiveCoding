using System.Runtime.InteropServices;

namespace PredictiveCoding.Device.Interop;

/// <summary>Direction flag for <see cref="NativeMethods.pcd_memcpy_async"/>.</summary>
public enum MemcpyKind { HostToDevice = 1, DeviceToHost = 2, DeviceToDevice = 3 }

/// <summary>Reduction op for NCCL-style collectives.</summary>
public enum ReduceOp { Sum = 0, Prod = 1, Max = 2, Min = 3 }

// ─────────────────────────────────────────────────────────────────────────────
// NativeMethods  —  P/Invoke bindings to the native bridge "pcdevice"
// (libpcdevice.so / pcdevice.dll). Every function returns a DeviceStatus int
// (0 = success). All pointers are opaque device/host addresses (IntPtr).
//
// The managed layer never calls these unless DeviceRuntime.CudaAvailable is true,
// so the binaries load and run on a CPU-only machine.
// ─────────────────────────────────────────────────────────────────────────────

internal static class NativeMethods
{
    private const string Lib = "pcdevice";
    private const CallingConvention Cdecl = CallingConvention.Cdecl;

    // ── Device / context ───────────────────────────────────────────────────────
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_cuda_device_count();
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_set_device(int deviceId);
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_device_synchronize();
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_init();
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_shutdown();

    // ── Memory ─────────────────────────────────────────────────────────────────
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_malloc_async(out IntPtr ptr, nuint bytes, IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_free_async(IntPtr ptr, IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_malloc(out IntPtr ptr, nuint bytes);
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_free(IntPtr ptr);
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_host_alloc(out IntPtr ptr, nuint bytes);   // pinned
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_host_free(IntPtr ptr);

    // ── Copies (sync + async) ───────────────────────────────────────────────────
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_memcpy_h2d(IntPtr dst, IntPtr src, nuint bytes);
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_memcpy_d2h(IntPtr dst, IntPtr src, nuint bytes);
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_memcpy_d2d(IntPtr dst, IntPtr src, nuint bytes);
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_memcpy_async(IntPtr dst, IntPtr src, nuint bytes, int kind, IntPtr stream);

    // ── Streams / events ───────────────────────────────────────────────────────
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_stream_create(out IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_stream_destroy(IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_stream_synchronize(IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_event_create(out IntPtr evt);
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_event_record(IntPtr evt, IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_event_synchronize(IntPtr evt);
    [DllImport(Lib, CallingConvention = Cdecl)] internal static extern int pcd_event_destroy(IntPtr evt);

    // ── GEMM (cuBLAS / cuBLASLt), row-major  C[m,n] = alpha·A[m,k]·B[k,n] + beta·C ──
    [DllImport(Lib, CallingConvention = Cdecl)]
    internal static extern int pcd_sgemm(int m, int n, int k, IntPtr a, IntPtr b, IntPtr c,
                                         float alpha, float beta, IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl)]
    internal static extern int pcd_sgemm_lt(int m, int n, int k, IntPtr a, IntPtr b, IntPtr c,
                                            float alpha, float beta, IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl)]
    internal static extern int pcd_sgemm_batched(int batch, int m, int n, int k,
                                                 IntPtr a, IntPtr b, IntPtr c,
                                                 float alpha, float beta, IntPtr stream);

    // ── NCCL-style collectives ───────────────────────────────────────────────────
    [DllImport(Lib, CallingConvention = Cdecl)]
    internal static extern int pcd_nccl_comm_init(int nDevices, int[] devices, [Out] IntPtr[] comms);
    [DllImport(Lib, CallingConvention = Cdecl)]
    internal static extern int pcd_nccl_comm_destroy(IntPtr comm);
    [DllImport(Lib, CallingConvention = Cdecl)]
    internal static extern int pcd_nccl_all_reduce(IntPtr comm, IntPtr send, IntPtr recv, nuint count, int op, IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl)]
    internal static extern int pcd_nccl_broadcast(IntPtr comm, IntPtr buf, nuint count, int root, IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl)]
    internal static extern int pcd_nccl_reduce_scatter(IntPtr comm, IntPtr send, IntPtr recv, nuint recvCount, int op, IntPtr stream);
    [DllImport(Lib, CallingConvention = Cdecl)]
    internal static extern int pcd_nccl_all_gather(IntPtr comm, IntPtr send, IntPtr recv, nuint sendCount, IntPtr stream);
}
