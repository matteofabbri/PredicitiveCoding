using System.Collections.Concurrent;
using PredictiveCoding.Device.Interop;

namespace PredictiveCoding.Device.Streams;

// ─────────────────────────────────────────────────────────────────────────────
// CudaStreamManager  —  device-aware stream / event scheduling (cudart).
//
//   • One default stream per calling thread (per device).
//   • Additional named streams per device for overlap.
//   • Events for fine-grained synchronisation.
//
// On a CPU-only machine every stream is the default (null) stream and the
// synchronisation calls are no-ops, so the API works unchanged.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class CudaStreamManager : IDisposable
{
    public static CudaStreamManager Shared { get; } = new();

    private readonly ThreadLocal<Dictionary<int, CudaStreamHandle>> _threadStreams =
        new(() => new Dictionary<int, CudaStreamHandle>());
    private readonly ConcurrentBag<CudaStreamHandle> _owned = new();

    public CudaStreamHandle ThreadStream(int deviceId)
    {
        if (!DeviceRuntime.CudaAvailable) return CudaStreamHandle.Default;

        var map = _threadStreams.Value!;
        if (!map.TryGetValue(deviceId, out var stream))
        {
            stream = CreateStream(deviceId);
            map[deviceId] = stream;
        }
        return stream;
    }

    public CudaStreamHandle CreateStream(int deviceId)
    {
        if (!DeviceRuntime.CudaAvailable) return CudaStreamHandle.Default;
        NativeCheck.Ok(Cuda.cudaSetDevice(deviceId), "cudaSetDevice");
        NativeCheck.Ok(Cuda.cudaStreamCreate(out IntPtr s), "cudaStreamCreate");
        var handle = new CudaStreamHandle(s);
        _owned.Add(handle);
        return handle;
    }

    public void Synchronize(CudaStreamHandle stream)
    {
        if (!DeviceRuntime.CudaAvailable) return;
        NativeCheck.Ok(Cuda.cudaStreamSynchronize(stream.Handle), "cudaStreamSynchronize");
    }

    public void SynchronizeDevice()
    {
        if (!DeviceRuntime.CudaAvailable) return;
        NativeCheck.Ok(Cuda.cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    }

    // ── Events ─────────────────────────────────────────────────────────────────

    public CudaEventHandle RecordEvent(CudaStreamHandle stream)
    {
        if (!DeviceRuntime.CudaAvailable) return default;
        NativeCheck.Ok(Cuda.cudaEventCreate(out IntPtr e), "cudaEventCreate");
        NativeCheck.Ok(Cuda.cudaEventRecord(e, stream.Handle), "cudaEventRecord");
        return new CudaEventHandle(e);
    }

    public void WaitEvent(CudaEventHandle evt)
    {
        if (evt.IsNull) return;
        NativeCheck.Ok(Cuda.cudaEventSynchronize(evt.Handle), "cudaEventSynchronize");
        Cuda.cudaEventDestroy(evt.Handle);
    }

    public void Dispose()
    {
        while (_owned.TryTake(out var s))
            if (!s.IsDefault) Cuda.cudaStreamDestroy(s.Handle);
        _threadStreams.Dispose();
    }
}
