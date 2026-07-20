using System.Runtime.InteropServices;
using PredictiveCoding.Device.Interop;
using PredictiveCoding.LLM.Infrastructure;

namespace PredictiveCoding.Device;

// ─────────────────────────────────────────────────────────────────────────────
// RocmGemmProvider  —  the AMD mirror of CudaGemmProvider (ROADMAP-Scale A4).
//
// Same architecture: plugs into LLM's IGemmProvider, calls hipBLAS directly
// through P/Invoke (standard ROCm libraries, no custom native DLL), computes
// the row-major product as the swapped column-major transpose, falls back to
// the CPU provider per call when no ROCm runtime/GPU is present.
//
// ⚠ UNTESTED-WITHOUT-HARDWARE: mirrors the verified CUDA path 1:1 (including
// hipBLAS's different operation enum, N=111/T=112) but has never run on an AMD
// GPU — no project machine has one. The CPU-fallback path IS covered by tests
// on every machine; the hip path awaits a supported Radeon. MatMulMixed
// forwards to the CPU oracle: the tensor-core-equivalent (hipblasGemmEx on
// Matrix Cores) is deliberately not mirrored blind.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class RocmGemmProvider : IGemmProvider
{
    private static readonly object _handleGate = new();
    private static readonly Dictionary<int, IntPtr> _handles = new();

    private readonly IGemmProvider _cpuFallback = new CpuGemmProvider();
    private readonly int _deviceId;

    public RocmGemmProvider(int deviceId = 0) => _deviceId = deviceId;

    public float[] MatMul(float[] a, int m, int k, float[] b, int n)
    {
        if (!DeviceRuntime.RocmAvailable) return _cpuFallback.MatMul(a, m, k, b, n);

        nuint aBytes = (nuint)(sizeof(float) * m * k);
        nuint bBytes = (nuint)(sizeof(float) * k * n);
        nuint cBytes = (nuint)(sizeof(float) * m * n);

        NativeCheck.Ok(Hip.hipSetDevice(_deviceId), "hipSetDevice");
        NativeCheck.Ok(Hip.hipMalloc(out IntPtr dA, aBytes), "hipMalloc A");
        NativeCheck.Ok(Hip.hipMalloc(out IntPtr dB, bBytes), "hipMalloc B");
        NativeCheck.Ok(Hip.hipMalloc(out IntPtr dC, cBytes), "hipMalloc C");
        try
        {
            Upload(a, dA, aBytes);
            Upload(b, dB, bBytes);

            IntPtr h = HandleFor(_deviceId);
            float alpha = 1f, beta = 0f;
            // Row-major C[m,n] = A·B ≡ column-major Cᵀ = Bᵀ·Aᵀ — same swap as
            // the CUDA provider, with hipBLAS's own op enum (N = 111).
            NativeCheck.Ok(Hipblas.hipblasSgemm(h, (int)HipblasOp.N, (int)HipblasOp.N, n, m, k,
                                                ref alpha, dB, n, dA, k, ref beta, dC, n), "hipblasSgemm");
            NativeCheck.Ok(Hip.hipDeviceSynchronize(), "hipDeviceSynchronize");

            var c = new float[m * n];
            Download(dC, c, cBytes);
            return c;
        }
        finally
        {
            Hip.hipFree(dA);
            Hip.hipFree(dB);
            Hip.hipFree(dC);
        }
    }

    /// <summary>Mixed path: CPU oracle — the Matrix-Core equivalent is not mirrored blind (see header).</summary>
    public float[] MatMulMixed(float[] a, int m, int k, float[] b, int n)
        => _cpuFallback.MatMulMixed(a, m, k, b, n);

    private static IntPtr HandleFor(int deviceId)
    {
        lock (_handleGate)
        {
            if (_handles.TryGetValue(deviceId, out IntPtr h)) return h;
            Hip.hipSetDevice(deviceId);
            NativeCheck.Ok(Hipblas.hipblasCreate(out h), "hipblasCreate");
            _handles[deviceId] = h;
            return h;
        }
    }

    private static void Upload(float[] host, IntPtr dst, nuint bytes)
    {
        var hh = GCHandle.Alloc(host, GCHandleType.Pinned);
        try { NativeCheck.Ok(Hip.hipMemcpy(dst, hh.AddrOfPinnedObject(), bytes, (int)MemcpyKind.HostToDevice), "hip h2d"); }
        finally { hh.Free(); }
    }

    private static void Download(IntPtr src, float[] host, nuint bytes)
    {
        var hh = GCHandle.Alloc(host, GCHandleType.Pinned);
        try { NativeCheck.Ok(Hip.hipMemcpy(hh.AddrOfPinnedObject(), src, bytes, (int)MemcpyKind.DeviceToHost), "hip d2h"); }
        finally { hh.Free(); }
    }
}
