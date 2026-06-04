using System.Runtime.InteropServices;
using PredictiveCoding.Device.Allocators;
using PredictiveCoding.Device.Interop;
using PredictiveCoding.Device.Streams;
using PredictiveCoding.LLM.Infrastructure;

namespace PredictiveCoding.Device;

// ─────────────────────────────────────────────────────────────────────────────
// CudaGemmProvider  —  GPU GEMM backend that plugs into LLM's IGemmProvider.
//
// Implements the same interface as CpuGemmProvider, so assigning it to
// Gemm.Provider transparently routes every matmul in the LLM stack through
// cuBLASLt. Row-major C[m,n] = A[m,k]·B[k,n] is handled by the native bridge
// (transpose flags). When CUDA is unavailable it forwards to the CPU provider.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class CudaGemmProvider : IGemmProvider
{
    private readonly IGemmProvider    _cpuFallback = new CpuGemmProvider();
    private readonly IDeviceAllocator _allocator   = CudaAllocator.Shared;
    private readonly int              _deviceId;

    public CudaGemmProvider(int deviceId = 0) => _deviceId = deviceId;

    public float[] MatMul(float[] a, int m, int k, float[] b, int n)
    {
        if (!DeviceRuntime.CudaAvailable) return _cpuFallback.MatMul(a, m, k, b, n);

        var stream = CudaStreamManager.Shared.ThreadStream(_deviceId);
        nuint aBytes = (nuint)(sizeof(float) * m * k);
        nuint bBytes = (nuint)(sizeof(float) * k * n);
        nuint cBytes = (nuint)(sizeof(float) * m * n);

        IntPtr dA = _allocator.Allocate(_deviceId, aBytes, stream);
        IntPtr dB = _allocator.Allocate(_deviceId, bBytes, stream);
        IntPtr dC = _allocator.Allocate(_deviceId, cBytes, stream);
        try
        {
            Upload(a, dA, aBytes);
            Upload(b, dB, bBytes);
            MatMulDevice(dA, dB, dC, m, k, n, stream);

            var c = new float[m * n];
            Download(dC, c, cBytes);
            CudaStreamManager.Shared.Synchronize(stream);
            return c;
        }
        finally
        {
            _allocator.Free(_deviceId, dA, aBytes, stream);
            _allocator.Free(_deviceId, dB, bBytes, stream);
            _allocator.Free(_deviceId, dC, cBytes, stream);
        }
    }

    /// <summary>Device-pointer GEMM (no host transfer) used by <see cref="TensorOps"/>.</summary>
    public void MatMulDevice(IntPtr dA, IntPtr dB, IntPtr dC, int m, int k, int n, CudaStreamHandle stream)
        => NativeCheck.Ok(NativeMethods.pcd_sgemm_lt(m, n, k, dA, dB, dC, 1f, 0f, stream.Handle), "sgemm_lt");

    /// <summary>Batched device-pointer GEMM (contiguous batch).</summary>
    public void MatMulBatchedDevice(IntPtr dA, IntPtr dB, IntPtr dC, int batch, int m, int k, int n, CudaStreamHandle stream)
        => NativeCheck.Ok(NativeMethods.pcd_sgemm_batched(batch, m, n, k, dA, dB, dC, 1f, 0f, stream.Handle), "sgemm_batched");

    // ── transfer helpers ────────────────────────────────────────────────────────

    private static void Upload(float[] host, IntPtr dst, nuint bytes)
    {
        var h = GCHandle.Alloc(host, GCHandleType.Pinned);
        try { NativeCheck.Ok(NativeMethods.pcd_memcpy_h2d(dst, h.AddrOfPinnedObject(), bytes), "h2d"); }
        finally { h.Free(); }
    }

    private static void Download(IntPtr src, float[] host, nuint bytes)
    {
        var h = GCHandle.Alloc(host, GCHandleType.Pinned);
        try { NativeCheck.Ok(NativeMethods.pcd_memcpy_d2h(h.AddrOfPinnedObject(), src, bytes), "d2h"); }
        finally { h.Free(); }
    }
}
