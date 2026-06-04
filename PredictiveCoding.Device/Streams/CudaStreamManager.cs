using System.Collections.Concurrent;
using PredictiveCoding.Device.Interop;

namespace PredictiveCoding.Device.Streams;

// ─────────────────────────────────────────────────────────────────────────────
// CudaStreamManager  —  device-aware stream / event scheduling.
//
//   • One default stream per calling thread (per device), so concurrent host
//     threads submit work on independent streams.
//   • Additional named streams can be created per device for overlap.
//   • Events allow fine-grained synchronisation between streams.
//
// On a CPU-only machine every stream is the default (null) stream and all
// operations are synchronous no-ops, so the API works unchanged.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class CudaStreamManager : IDisposable
{
    public static CudaStreamManager Shared { get; } = new();

    private readonly ThreadLocal<Dictionary<int, CudaStreamHandle>> _threadStreams =
        new(() => new Dictionary<int, CudaStreamHandle>());
    private readonly ConcurrentBag<CudaStreamHandle> _owned = new();

    /// <summary>The current thread's default stream for <paramref name="deviceId"/>.</summary>
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

    /// <summary>Creates a new CUDA stream on the given device (caller may keep it for overlap).</summary>
    public CudaStreamHandle CreateStream(int deviceId)
    {
        if (!DeviceRuntime.CudaAvailable) return CudaStreamHandle.Default;
        NativeCheck.Ok(NativeMethods.pcd_set_device(deviceId), "set_device");
        NativeCheck.Ok(NativeMethods.pcd_stream_create(out IntPtr s), "stream_create");
        var handle = new CudaStreamHandle(s);
        _owned.Add(handle);
        return handle;
    }

    /// <summary>Blocks the host until all work on <paramref name="stream"/> has completed.</summary>
    public void Synchronize(CudaStreamHandle stream)
    {
        if (!DeviceRuntime.CudaAvailable) return;
        NativeCheck.Ok(NativeMethods.pcd_stream_synchronize(stream.Handle), "stream_synchronize");
    }

    /// <summary>Blocks the host until every device operation has completed.</summary>
    public void SynchronizeDevice()
    {
        if (!DeviceRuntime.CudaAvailable) return;
        NativeCheck.Ok(NativeMethods.pcd_device_synchronize(), "device_synchronize");
    }

    // ── Events ─────────────────────────────────────────────────────────────────

    public CudaEventHandle RecordEvent(CudaStreamHandle stream)
    {
        if (!DeviceRuntime.CudaAvailable) return default;
        NativeCheck.Ok(NativeMethods.pcd_event_create(out IntPtr e), "event_create");
        NativeCheck.Ok(NativeMethods.pcd_event_record(e, stream.Handle), "event_record");
        return new CudaEventHandle(e);
    }

    public void WaitEvent(CudaEventHandle evt)
    {
        if (evt.IsNull) return;
        NativeCheck.Ok(NativeMethods.pcd_event_synchronize(evt.Handle), "event_synchronize");
        NativeMethods.pcd_event_destroy(evt.Handle);
    }

    public void Dispose()
    {
        while (_owned.TryTake(out var s))
            if (!s.IsDefault) NativeMethods.pcd_stream_destroy(s.Handle);
        _threadStreams.Dispose();
    }
}
