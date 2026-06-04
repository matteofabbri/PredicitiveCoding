using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using PredictiveCoding.Core;
using PredictiveCoding.Device.Allocators;
using PredictiveCoding.Device.Interop;
using PredictiveCoding.Device.Streams;

namespace PredictiveCoding.Device;

// ─────────────────────────────────────────────────────────────────────────────
// DeviceTensor<T>  —  a tensor whose data lives on a chosen device.
//
// Wraps the project's host tensor (DenseTensor<T>, itself built on
// System.Numerics.Tensors.TensorPrimitives — S.N.T is never modified) and an
// optional device buffer. PyTorch-style placement: .To(Device.Cuda()) /
// .To(Device.Cpu). When CUDA is unavailable every placement resolves to the CPU
// and the device buffer is simply never allocated.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class DeviceTensor<T> : IDisposable where T : unmanaged, System.Numerics.INumber<T>
{
    private readonly int[]            _shape;
    private readonly IDeviceAllocator _allocator;

    /// <summary>Host-side tensor (null when the data currently lives only on a device).</summary>
    public DenseTensor<T>? HostTensor { get; private set; }

    /// <summary>Raw device pointer (IntPtr.Zero when on CPU).</summary>
    public IntPtr DevicePtr { get; private set; }

    public DeviceKind Device  { get; private set; }
    public int        DeviceId { get; private set; }
    public int        ElementCount { get; }
    public int        SizeInBytes  => ElementCount * Unsafe.SizeOf<T>();

    /// <summary>Stream this tensor's transfers/kernels are issued on.</summary>
    public CudaStreamHandle Stream { get; set; } = CudaStreamHandle.Default;

    public bool IsOnDevice => DevicePtr != IntPtr.Zero && Device != DeviceKind.Cpu;

    private DeviceTensor(int[] shape, DenseTensor<T>? host, IDeviceAllocator allocator)
    {
        _shape       = (int[])shape.Clone();
        _allocator   = allocator;
        HostTensor   = host;
        Device       = DeviceKind.Cpu;
        ElementCount = 1;
        foreach (int d in shape) ElementCount *= d;
    }

    // ── Construction ───────────────────────────────────────────────────────────

    /// <summary>Wraps a host tensor (stays on the CPU until moved).</summary>
    public static DeviceTensor<T> FromHost(DenseTensor<T> host, IDeviceAllocator? allocator = null)
        => new(host.Dimensions, host, allocator ?? CudaAllocator.Shared);

    // ── Placement ──────────────────────────────────────────────────────────────

    /// <summary>Returns a copy of this tensor placed on <paramref name="target"/> (CUDA→CPU fallback automatic).</summary>
    public DeviceTensor<T> To(Device target)
    {
        var resolved = DeviceRuntime.Resolve(target);

        if (resolved.IsCpu)
            return new DeviceTensor<T>(_shape, ToHost(), _allocator) { Device = DeviceKind.Cpu };

        // → CUDA (single or one of multi).
        int deviceId = resolved.Kind == DeviceKind.CudaMulti ? 0 : resolved.Id;
        var stream   = CudaStreamManager.Shared.ThreadStream(deviceId);

        var host = ToHost();   // ensure we have host bytes to upload
        var gpu  = new DeviceTensor<T>(_shape, null, _allocator)
        {
            Device = DeviceKind.Cuda, DeviceId = deviceId, Stream = stream,
        };
        gpu.DevicePtr = _allocator.Allocate(deviceId, (nuint)gpu.SizeInBytes, stream);
        gpu.UploadFrom(host, stream, async: false);
        return gpu;
    }

    /// <summary>Materialises the data on the host (downloading from the device if needed).</summary>
    public DenseTensor<T> ToHost()
    {
        if (HostTensor is not null) return HostTensor;

        var host = new DenseTensor<T>(_shape);
        Download(host, Stream, async: false);
        HostTensor = host;
        return host;
    }

    // ── Transfers ──────────────────────────────────────────────────────────────

    internal void UploadFrom(DenseTensor<T> host, CudaStreamHandle stream, bool async)
    {
        var handle = GCHandle.Alloc(host.Buffer, GCHandleType.Pinned);
        try
        {
            IntPtr src = handle.AddrOfPinnedObject();
            if (async)
                NativeCheck.Ok(NativeMethods.pcd_memcpy_async(DevicePtr, src, (nuint)SizeInBytes, (int)MemcpyKind.HostToDevice, stream.Handle), "h2d_async");
            else
                NativeCheck.Ok(NativeMethods.pcd_memcpy_h2d(DevicePtr, src, (nuint)SizeInBytes), "h2d");
        }
        finally { handle.Free(); }
    }

    internal void Download(DenseTensor<T> host, CudaStreamHandle stream, bool async)
    {
        var handle = GCHandle.Alloc(host.Buffer, GCHandleType.Pinned);
        try
        {
            IntPtr dst = handle.AddrOfPinnedObject();
            if (async)
                NativeCheck.Ok(NativeMethods.pcd_memcpy_async(dst, DevicePtr, (nuint)SizeInBytes, (int)MemcpyKind.DeviceToHost, stream.Handle), "d2h_async");
            else
                NativeCheck.Ok(NativeMethods.pcd_memcpy_d2h(dst, DevicePtr, (nuint)SizeInBytes), "d2h");
        }
        finally { handle.Free(); }
    }

    /// <summary>Allocates an uninitialised tensor with the same shape on the same device.</summary>
    public DeviceTensor<T> EmptyLike()
    {
        var t = new DeviceTensor<T>(_shape, null, _allocator) { Device = Device, DeviceId = DeviceId, Stream = Stream };
        if (IsOnDevice) t.DevicePtr = _allocator.Allocate(DeviceId, (nuint)t.SizeInBytes, Stream);
        else            t.HostTensor = new DenseTensor<T>(_shape);
        return t;
    }

    public int[] Shape => (int[])_shape.Clone();

    public void Dispose()
    {
        if (DevicePtr != IntPtr.Zero)
        {
            _allocator.Free(DeviceId, DevicePtr, (nuint)SizeInBytes, Stream);
            DevicePtr = IntPtr.Zero;
        }
        HostTensor = null;
    }
}
