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
// ─────────────────────────────────────────────────────────────────────────────

public sealed class CudaAllocator : IDeviceAllocator
{
    /// <summary>Process-wide default allocator.</summary>
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

        NativeCheck.Ok(NativeMethods.pcd_set_device(deviceId), "set_device");

        int status = NativeMethods.pcd_malloc_async(out IntPtr ptr, bucket, stream.Handle);
        if (status != 0)
            NativeCheck.Ok(NativeMethods.pcd_malloc(out ptr, bucket), "malloc(fallback)"); // sync fallback
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
        // Pool full → actually release.
        if (NativeMethods.pcd_free_async(ptr, stream.Handle) != 0)
            NativeMethods.pcd_free(ptr);
    }

    // ── Pinned host memory ────────────────────────────────────────────────────

    public IntPtr AllocatePinnedHost(nuint bytes)
    {
        nuint bucket = RoundUp(bytes);
        lock (_gate)
        {
            if (_pinnedPool.TryGetValue(bucket, out var pool) && pool.Count > 0) return pool.Pop();
        }
        NativeCheck.Ok(NativeMethods.pcd_host_alloc(out IntPtr ptr, bucket), "host_alloc");
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
        NativeMethods.pcd_host_free(ptr);
    }

    /// <summary>Releases every pooled buffer back to the driver.</summary>
    public void TrimPools()
    {
        lock (_gate)
        {
            foreach (var pool in _devicePool.Values) while (pool.Count > 0) NativeMethods.pcd_free(pool.Pop());
            foreach (var pool in _pinnedPool.Values) while (pool.Count > 0) NativeMethods.pcd_host_free(pool.Pop());
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
