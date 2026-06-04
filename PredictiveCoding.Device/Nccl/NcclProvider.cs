using System.Runtime.InteropServices;
using PredictiveCoding.Device.Allocators;
using PredictiveCoding.Device.Interop;
using PredictiveCoding.Device.Streams;

namespace PredictiveCoding.Device.Nccl;

// ─────────────────────────────────────────────────────────────────────────────
// NcclProvider  —  NCCL-style collectives across a set of devices.
//
//   AllReduce / Broadcast / ReduceScatter / AllGather, plus communicator
//   management. When ≥2 CUDA devices are present a real NCCL communicator per
//   device is created and AllReduce/Broadcast are issued through it; otherwise an
//   equivalent CPU implementation (the "mock NCCL") runs so the API works on any
//   machine. Buffers are per-rank host arrays (one float[] per device).
// ─────────────────────────────────────────────────────────────────────────────

public interface INcclProvider : IDisposable
{
    int RankCount { get; }
    void AllReduce(float[][] buffers, ReduceOp op = ReduceOp.Sum);
    void Broadcast(float[][] buffers, int root);
    void ReduceScatter(float[][] send, float[][] recv, ReduceOp op = ReduceOp.Sum);
    void AllGather(float[][] send, float[][] recv);
}

public sealed class NcclProvider : INcclProvider
{
    private readonly int[]             _devices;
    private readonly NcclCommHandle[]  _comms;
    private readonly bool              _useNccl;

    public int RankCount => _devices.Length;
    public bool UsesNccl => _useNccl;

    public NcclProvider(int[] devices)
    {
        if (devices.Length < 1) throw new ArgumentException("At least one device is required.", nameof(devices));
        _devices = (int[])devices.Clone();
        _useNccl = DeviceRuntime.MultiGpuAvailable && devices.Length > 1;

        if (_useNccl)
        {
            var raw = new IntPtr[devices.Length];
            NativeCheck.Ok(NativeMethods.pcd_nccl_comm_init(devices.Length, _devices, raw), "nccl_comm_init");
            _comms = raw.Select(p => new NcclCommHandle(p)).ToArray();
        }
        else _comms = [];
    }

    /// <summary>Creates a provider over all available CUDA devices (or a single CPU rank).</summary>
    public static NcclProvider ForAllDevices()
    {
        int n = Math.Max(1, DeviceRuntime.CudaDeviceCount);
        return new NcclProvider(Enumerable.Range(0, n).ToArray());
    }

    // ── Collectives ────────────────────────────────────────────────────────────

    public void AllReduce(float[][] buffers, ReduceOp op = ReduceOp.Sum)
    {
        ValidateUniform(buffers);
        if (_useNccl) { GpuAllReduce(buffers, op); return; }

        var reduced = (float[])buffers[0].Clone();
        for (int r = 1; r < RankCount; r++) Combine(reduced, buffers[r], op);
        foreach (var buf in buffers) Array.Copy(reduced, buf, reduced.Length);
    }

    public void Broadcast(float[][] buffers, int root)
    {
        ValidateUniform(buffers);
        if (root < 0 || root >= RankCount) throw new ArgumentOutOfRangeException(nameof(root));
        if (_useNccl) { GpuBroadcast(buffers, root); return; }

        foreach (var buf in buffers) Array.Copy(buffers[root], buf, buf.Length);
    }

    public void ReduceScatter(float[][] send, float[][] recv, ReduceOp op = ReduceOp.Sum)
    {
        ValidateUniform(send);
        int total = send[0].Length;
        if (total % RankCount != 0) throw new ArgumentException("Send length must divide evenly across ranks.");
        int chunk = total / RankCount;

        var reduced = (float[])send[0].Clone();
        for (int r = 1; r < RankCount; r++) Combine(reduced, send[r], op);
        for (int r = 0; r < RankCount; r++)
        {
            if (recv[r].Length != chunk) throw new ArgumentException($"recv[{r}] must have length {chunk}.");
            Array.Copy(reduced, r * chunk, recv[r], 0, chunk);
        }
    }

    public void AllGather(float[][] send, float[][] recv)
    {
        ValidateUniform(send);
        int len = send[0].Length;
        for (int r = 0; r < RankCount; r++)
        {
            if (recv[r].Length != len * RankCount) throw new ArgumentException($"recv[{r}] must have length {len * RankCount}.");
            for (int s = 0; s < RankCount; s++) Array.Copy(send[s], 0, recv[r], s * len, len);
        }
    }

    // ── GPU (NCCL) paths ─────────────────────────────────────────────────────

    private void GpuAllReduce(float[][] buffers, ReduceOp op)
    {
        int count = buffers[0].Length;
        nuint bytes = (nuint)(sizeof(float) * count);
        for (int r = 0; r < RankCount; r++)
        {
            int dev = _devices[r];
            var stream = CudaStreamManager.Shared.ThreadStream(dev);
            IntPtr buf = CudaAllocator.Shared.Allocate(dev, bytes, stream);
            try
            {
                Upload(buffers[r], buf, bytes);
                NativeCheck.Ok(NativeMethods.pcd_nccl_all_reduce(_comms[r].Handle, buf, buf, (nuint)count, (int)op, stream.Handle), "nccl_all_reduce");
                CudaStreamManager.Shared.Synchronize(stream);
                Download(buf, buffers[r], bytes);
            }
            finally { CudaAllocator.Shared.Free(dev, buf, bytes, stream); }
        }
    }

    private void GpuBroadcast(float[][] buffers, int root)
    {
        int count = buffers[0].Length;
        nuint bytes = (nuint)(sizeof(float) * count);
        for (int r = 0; r < RankCount; r++)
        {
            int dev = _devices[r];
            var stream = CudaStreamManager.Shared.ThreadStream(dev);
            IntPtr buf = CudaAllocator.Shared.Allocate(dev, bytes, stream);
            try
            {
                Upload(buffers[r], buf, bytes);
                NativeCheck.Ok(NativeMethods.pcd_nccl_broadcast(_comms[r].Handle, buf, (nuint)count, root, stream.Handle), "nccl_broadcast");
                CudaStreamManager.Shared.Synchronize(stream);
                Download(buf, buffers[r], bytes);
            }
            finally { CudaAllocator.Shared.Free(dev, buf, bytes, stream); }
        }
    }

    // ── helpers ────────────────────────────────────────────────────────────────

    private void ValidateUniform(float[][] buffers)
    {
        if (buffers.Length != RankCount) throw new ArgumentException($"Expected {RankCount} per-rank buffers.");
        int len = buffers[0].Length;
        foreach (var b in buffers) if (b.Length != len) throw new ArgumentException("All rank buffers must be the same length.");
    }

    private static void Combine(float[] acc, float[] other, ReduceOp op)
    {
        for (int i = 0; i < acc.Length; i++)
            acc[i] = op switch
            {
                ReduceOp.Sum  => acc[i] + other[i],
                ReduceOp.Prod => acc[i] * other[i],
                ReduceOp.Max  => MathF.Max(acc[i], other[i]),
                ReduceOp.Min  => MathF.Min(acc[i], other[i]),
                _             => acc[i] + other[i],
            };
    }

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

    public void Dispose()
    {
        foreach (var c in _comms) if (!c.IsNull) NativeMethods.pcd_nccl_comm_destroy(c.Handle);
    }
}
