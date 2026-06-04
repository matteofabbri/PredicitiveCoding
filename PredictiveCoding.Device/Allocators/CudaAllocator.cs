using PredictiveCoding.Device.Interop;

namespace PredictiveCoding.Device.Allocators;

// ─────────────────────────────────────────────────────────────────────────────
// CudaAllocator  —  pooling, stream-ordered, pinned-host device allocator.
//
//   • Device buffers are bucketed by a power-of-two rounded size; freed buffers
//     return to a per-(device, bucket) free list and are reused, avoiding the
//     cost of repeated cudaMalloc.
//   • Allocation uses cudaMallocAsync (stream-ordered); if that fails it falls
//     back to plain cudaMalloc.
//   • Pinned host buffers (cudaHostAlloc) are pooled the same way for fast,
//     overlappable H2D / D2H transfers.
//
// Calls go straight to the CUDA runtime (cudart) — no custom native library.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class CudaAllocator : IDeviceAllocator
{
    public static CudaAllocator Shared { get; } = new();

    private const int MaxPooledPerBucket = 32;
    private const int MinBucket          = 256;

    private readonly object _gate = new();
    private readonly Dictionary<(int dev, nuint size), Stack<IntPtr>> _devicePool = new();
    private readonly Dictionary<nuint, Stack<IntPtr>>                 _pinnedPool = new();

    // ── Device memory ────────────────────────────────────────────────────────

    public IntPtr Allocate(int deviceId, nuint bytes, CudaStreamHandle stream)
    {
        nuint bucket = RoundUp(bytes);
        lock (_gate)
        {
            if (_devicePool.TryGetValue((deviceId, bucket), out var pool) && pool.Count > 0)
                return pool.Pop();
        }

        NativeCheck.Ok(Cuda.cudaSetDevice(deviceId), "cudaSetDevice");

        if (Cuda.cudaMallocAsync(out IntPtr ptr, bucket, stream.Handle) != 0)
            NativeCheck.Ok(Cuda.cudaMalloc(out ptr, bucket), "cudaMalloc(fallback)");
        return ptr;
    }

    public void Free(int deviceId, IntPtr ptr, nuint bytes, CudaStreamHandle stream)
    {
        if (ptr == IntPtr.Zero) return;
        nuint bucket = RoundUp(bytes);
        lock (_gate)
        {
            var key = (deviceId, bucket);
            if (!_devicePool.TryGetValue(key, out var pool)) _devicePool[key] = pool = new Stack<IntPtr>();
            if (pool.Count < MaxPooledPerBucket) { pool.Push(ptr); return; }
        }
        if (Cuda.cudaFreeAsync(ptr, stream.Handle) != 0) Cuda.cudaFree(ptr);
    }

    // ── Pinned host memory ────────────────────────────────────────────────────

    public IntPtr AllocatePinnedHost(nuint bytes)
    {
        nuint bucket = RoundUp(bytes);
        lock (_gate)
        {
            if (_pinnedPool.TryGetValue(bucket, out var pool) && pool.Count > 0) return pool.Pop();
        }
        NativeCheck.Ok(Cuda.cudaHostAlloc(out IntPtr ptr, bucket), "cudaHostAlloc");
        return ptr;
    }

    public void FreePinnedHost(IntPtr ptr, nuint bytes)
    {
        if (ptr == IntPtr.Zero) return;
        nuint bucket = RoundUp(bytes);
        lock (_gate)
        {
            if (!_pinnedPool.TryGetValue(bucket, out var pool)) _pinnedPool[bucket] = pool = new Stack<IntPtr>();
            if (pool.Count < MaxPooledPerBucket) { pool.Push(ptr); return; }
        }
        Cuda.cudaFreeHost(ptr);
    }

    /// <summary>Releases every pooled buffer back to the driver.</summary>
    public void TrimPools()
    {
        lock (_gate)
        {
            foreach (var pool in _devicePool.Values) while (pool.Count > 0) Cuda.cudaFree(pool.Pop());
            foreach (var pool in _pinnedPool.Values) while (pool.Count > 0) Cuda.cudaFreeHost(pool.Pop());
            _devicePool.Clear();
            _pinnedPool.Clear();
        }
    }

    private static nuint RoundUp(nuint bytes)
    {
        nuint v = MinBucket;
        while (v < bytes) v <<= 1;
        return v;
    }
}
