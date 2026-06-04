using PredictiveCoding.Device.Interop;

namespace PredictiveCoding.Device.Allocators;

/// <summary>Device memory allocator (device buffers + pinned host buffers).</summary>
public interface IDeviceAllocator
{
    /// <summary>Allocates <paramref name="bytes"/> of device memory on the given stream.</summary>
    IntPtr Allocate(int deviceId, nuint bytes, CudaStreamHandle stream);

    /// <summary>Returns a device buffer (may be pooled rather than freed).</summary>
    void Free(int deviceId, IntPtr ptr, nuint bytes, CudaStreamHandle stream);

    /// <summary>Allocates page-locked (pinned) host memory for fast async copies.</summary>
    IntPtr AllocatePinnedHost(nuint bytes);

    void FreePinnedHost(IntPtr ptr, nuint bytes);
}
