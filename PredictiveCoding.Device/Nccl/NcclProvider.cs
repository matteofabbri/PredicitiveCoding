namespace PredictiveCoding.Device.Nccl;

/// <summary>Reduction op for collectives.</summary>
public enum ReduceOp { Sum = 0, Prod = 1, Max = 2, Min = 3 }

// ─────────────────────────────────────────────────────────────────────────────
// NcclProvider  —  NCCL-style collectives across a set of devices.
//
//   AllReduce / Broadcast / ReduceScatter / AllGather over per-rank buffers
//   (one float[] per device). The algorithms run on the host (the "mock NCCL"),
//   which keeps the API working on any machine.
//
//   Real GPU NCCL is NOT part of cudart/cublas, so it is intentionally not wired
//   here (it would need its own libnccl binding) — a clean extension point that
//   doesn't reintroduce a custom native build.
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
    private readonly int[] _devices;

    public int RankCount => _devices.Length;

    public NcclProvider(int[] devices)
    {
        if (devices.Length < 1) throw new ArgumentException("At least one device is required.", nameof(devices));
        _devices = (int[])devices.Clone();
    }

    /// <summary>Creates a provider over all available CUDA devices (or a single CPU rank).</summary>
    public static NcclProvider ForAllDevices()
    {
        int n = Math.Max(1, DeviceRuntime.CudaDeviceCount);
        return new NcclProvider(Enumerable.Range(0, n).ToArray());
    }

    public void AllReduce(float[][] buffers, ReduceOp op = ReduceOp.Sum)
    {
        ValidateUniform(buffers);
        var reduced = (float[])buffers[0].Clone();
        for (int r = 1; r < RankCount; r++) Combine(reduced, buffers[r], op);
        foreach (var buf in buffers) Array.Copy(reduced, buf, reduced.Length);
    }

    public void Broadcast(float[][] buffers, int root)
    {
        ValidateUniform(buffers);
        if (root < 0 || root >= RankCount) throw new ArgumentOutOfRangeException(nameof(root));
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

    public void Dispose() { }
}
