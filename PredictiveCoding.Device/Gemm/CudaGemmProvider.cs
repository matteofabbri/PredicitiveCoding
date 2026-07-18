using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using PredictiveCoding.Device.Allocators;
using PredictiveCoding.Device.Interop;
using PredictiveCoding.Device.Streams;
using PredictiveCoding.LLM.Infrastructure;

namespace PredictiveCoding.Device;

// ─────────────────────────────────────────────────────────────────────────────
// CudaGemmProvider  —  GPU GEMM backend that plugs into LLM's IGemmProvider.
//
// Calls cuBLAS (cublasSgemm) directly through P/Invoke — no custom native DLL.
// cuBLAS is column-major, so row-major C[m,n] = A[m,k]·B[k,n] is computed as the
// transposed product Cᵀ = Bᵀ·Aᵀ (operands swapped). When CUDA is unavailable it
// forwards to the CPU provider.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class CudaGemmProvider : IGemmProvider
{
    private static readonly object _handleGate = new();
    private static readonly Dictionary<int, IntPtr> _handles = new();

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
    {
        IntPtr h = HandleFor(_deviceId);
        NativeCheck.Ok(Cuda.cudaSetDevice(_deviceId), "cudaSetDevice");
        Cublas.cublasSetStream_v2(h, stream.Handle);

        float alpha = 1f, beta = 0f;
        // Row-major C[m,n] = A·B  ≡  column-major Cᵀ = Bᵀ·Aᵀ : sgemm(N,N, n,m,k, B,n, A,k, C,n)
        NativeCheck.Ok(Cublas.cublasSgemm_v2(h, (int)CublasOp.N, (int)CublasOp.N, n, m, k,
                                             ref alpha, dB, n, dA, k, ref beta, dC, n), "cublasSgemm");
    }

    /// <summary>Batched (strided) device-pointer GEMM.</summary>
    public void MatMulBatchedDevice(IntPtr dA, IntPtr dB, IntPtr dC, int batch, int m, int k, int n, CudaStreamHandle stream)
    {
        IntPtr h = HandleFor(_deviceId);
        NativeCheck.Ok(Cuda.cudaSetDevice(_deviceId), "cudaSetDevice");
        Cublas.cublasSetStream_v2(h, stream.Handle);

        float alpha = 1f, beta = 0f;
        NativeCheck.Ok(Cublas.cublasSgemmStridedBatched(
            h, (int)CublasOp.N, (int)CublasOp.N, n, m, k,
            ref alpha, dB, n, (long)n * k, dA, k, (long)m * k,
            ref beta, dC, n, (long)m * n, batch), "cublasSgemmStridedBatched");
    }

    // ── Mixed precision (16-bit storage, 32-bit accumulation on tensor cores) ────

    /// <summary>
    /// Mixed-precision GEMM: operands are quantised to Half (16-bit) and multiplied
    /// on the tensor cores via <c>cublasGemmEx</c> with float32 accumulation; the
    /// result C is float32. Falls back to the CPU mixed path when CUDA is absent.
    /// </summary>
    public float[] MatMulMixed(float[] a, int m, int k, float[] b, int n)
    {
        if (!DeviceRuntime.CudaAvailable) return _cpuFallback.MatMulMixed(a, m, k, b, n);

        var aH = new Half[m * k]; TensorPrimitives.ConvertToHalf(a, aH);
        var bH = new Half[k * n]; TensorPrimitives.ConvertToHalf(b, bH);

        var stream = CudaStreamManager.Shared.ThreadStream(_deviceId);
        nuint aBytes = (nuint)(2 * m * k);              // Half = 2 bytes
        nuint bBytes = (nuint)(2 * k * n);
        nuint cBytes = (nuint)(sizeof(float) * m * n);  // C accumulated in float32

        IntPtr dA = _allocator.Allocate(_deviceId, aBytes, stream);
        IntPtr dB = _allocator.Allocate(_deviceId, bBytes, stream);
        IntPtr dC = _allocator.Allocate(_deviceId, cBytes, stream);
        try
        {
            UploadHalf(aH, dA, aBytes);
            UploadHalf(bH, dB, bBytes);
            MatMulMixedDevice(dA, dB, dC, m, k, n, stream);

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

    /// <summary>Device-pointer mixed GEMM (A,B in Half; C in float32) via cuBLAS tensor cores.</summary>
    public void MatMulMixedDevice(IntPtr dA, IntPtr dB, IntPtr dC, int m, int k, int n, CudaStreamHandle stream)
    {
        IntPtr h = HandleFor(_deviceId);
        NativeCheck.Ok(Cuda.cudaSetDevice(_deviceId), "cudaSetDevice");
        Cublas.cublasSetStream_v2(h, stream.Handle);

        float alpha = 1f, beta = 0f;
        // Row-major C[m,n] = A·B  ≡  column-major Cᵀ = Bᵀ·Aᵀ (operands swapped).
        // A,B are R16F; C is R32F; accumulation is Compute32F → tensor cores.
        NativeCheck.Ok(Cublas.cublasGemmEx(
            h, (int)CublasOp.N, (int)CublasOp.N, n, m, k,
            ref alpha,
            dB, (int)CudaDataType.R16F, n,
            dA, (int)CudaDataType.R16F, k,
            ref beta,
            dC, (int)CudaDataType.R32F, n,
            (int)CublasComputeType.Compute32F, (int)CublasGemmAlgo.Default), "cublasGemmEx");
    }

    private static void UploadHalf(Half[] host, IntPtr dst, nuint bytes)
    {
        var hh = GCHandle.Alloc(host, GCHandleType.Pinned);
        try { NativeCheck.Ok(Cuda.cudaMemcpy(dst, hh.AddrOfPinnedObject(), bytes, (int)MemcpyKind.HostToDevice), "h2d"); }
        finally { hh.Free(); }
    }

    private static IntPtr HandleFor(int deviceId)
    {
        lock (_handleGate)
        {
            if (_handles.TryGetValue(deviceId, out IntPtr h)) return h;
            Cuda.cudaSetDevice(deviceId);
            NativeCheck.Ok(Cublas.cublasCreate_v2(out h), "cublasCreate");
            _handles[deviceId] = h;
            return h;
        }
    }

    // ── transfer helpers ────────────────────────────────────────────────────────

    private static void Upload(float[] host, IntPtr dst, nuint bytes)
    {
        var hh = GCHandle.Alloc(host, GCHandleType.Pinned);
        try { NativeCheck.Ok(Cuda.cudaMemcpy(dst, hh.AddrOfPinnedObject(), bytes, (int)MemcpyKind.HostToDevice), "h2d"); }
        finally { hh.Free(); }
    }

    private static void Download(IntPtr src, float[] host, nuint bytes)
    {
        var hh = GCHandle.Alloc(host, GCHandleType.Pinned);
        try { NativeCheck.Ok(Cuda.cudaMemcpy(hh.AddrOfPinnedObject(), src, bytes, (int)MemcpyKind.DeviceToHost), "d2h"); }
        finally { hh.Free(); }
    }
}
