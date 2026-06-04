using PredictiveCoding.Core;

namespace PredictiveCoding.Device;

// ─────────────────────────────────────────────────────────────────────────────
// TensorExtensions  —  PyTorch-style placement helpers.
//
//   var g = host.ToDevice(Device.Cuda());     // DenseTensor → DeviceTensor on GPU
//   var h = g.To(Device.Cpu).ToHost();         // back to a host DenseTensor
// ─────────────────────────────────────────────────────────────────────────────

public static class TensorExtensions
{
    /// <summary>Wraps a host tensor as a CPU <see cref="DeviceTensor{T}"/>.</summary>
    public static DeviceTensor<T> AsDeviceTensor<T>(this DenseTensor<T> host)
        where T : unmanaged, System.Numerics.INumber<T>
        => DeviceTensor<T>.FromHost(host);

    /// <summary>Wraps a host tensor and immediately places it on <paramref name="device"/>.</summary>
    public static DeviceTensor<T> ToDevice<T>(this DenseTensor<T> host, Device device)
        where T : unmanaged, System.Numerics.INumber<T>
        => DeviceTensor<T>.FromHost(host).To(device);

    /// <summary>Downloads a device tensor back to a host <see cref="DenseTensor{T}"/>.</summary>
    public static DenseTensor<T> ToHostTensor<T>(this DeviceTensor<T> tensor)
        where T : unmanaged, System.Numerics.INumber<T>
        => tensor.To(Device.Cpu).ToHost();
}
