using System.Runtime.InteropServices;
using PredictiveCoding.Core;
using PredictiveCoding.Device.Interop;
using PredictiveCoding.Device.Streams;
using LlmGemm = PredictiveCoding.LLM.Infrastructure.Gemm;

namespace PredictiveCoding.Device;

// ─────────────────────────────────────────────────────────────────────────────
// TensorOps  —  device-aware tensor operations (PyTorch-like, sync + async).
//
//   MatMul / MatMulAsync run on the GPU (cuBLASLt) when both operands are
//   resident on a CUDA device, otherwise on the CPU. CopyTo / CopyToAsync move
//   data device↔device (or via host) and Synchronize blocks on a stream.
// ─────────────────────────────────────────────────────────────────────────────

public static class TensorOps
{
    private static readonly CudaGemmProvider Gpu = new();

    // ── MatMul ──────────────────────────────────────────────────────────────────

    /// <summary>Matrix multiply C[m,n] = A[m,k] · B[k,n] (synchronous).</summary>
    public static DeviceTensor<float> MatMul(DeviceTensor<float> a, DeviceTensor<float> b)
    {
        var (m, k, n) = Dims(a, b);

        if (a.IsOnDevice && b.IsOnDevice && DeviceRuntime.CudaAvailable)
        {
            var c = NewDeviceResult(m, n, a.DeviceId, a.Stream);
            Gpu.MatMulDevice(a.DevicePtr, b.DevicePtr, c.DevicePtr, m, k, n, a.Stream);
            CudaStreamManager.Shared.Synchronize(a.Stream);
            return c;
        }

        var cArr = LlmGemm.MatMul(a.ToHost().Buffer, m, k, b.ToHost().Buffer, n);
        return DeviceTensor<float>.FromHost(new DenseTensor<float>(cArr, m, n));
    }

    /// <summary>Matrix multiply on a CUDA stream without blocking; call <see cref="Synchronize"/> before reading.</summary>
    public static DeviceTensor<float> MatMulAsync(DeviceTensor<float> a, DeviceTensor<float> b, CudaStreamHandle stream)
    {
        var (m, k, n) = Dims(a, b);

        if (a.IsOnDevice && b.IsOnDevice && DeviceRuntime.CudaAvailable)
        {
            var c = NewDeviceResult(m, n, a.DeviceId, stream);
            Gpu.MatMulDevice(a.DevicePtr, b.DevicePtr, c.DevicePtr, m, k, n, stream);   // queued on stream
            return c;
        }

        var cArr = LlmGemm.MatMul(a.ToHost().Buffer, m, k, b.ToHost().Buffer, n);
        return DeviceTensor<float>.FromHost(new DenseTensor<float>(cArr, m, n));
    }

    // ── Copies ──────────────────────────────────────────────────────────────────

    /// <summary>Copies the contents of <paramref name="src"/> into <paramref name="dst"/> (synchronous).</summary>
    public static void CopyTo(DeviceTensor<float> src, DeviceTensor<float> dst)
    {
        if (src.SizeInBytes != dst.SizeInBytes) throw new ArgumentException("Tensors differ in size.");

        if (src.IsOnDevice && dst.IsOnDevice)
        {
            NativeCheck.Ok(Cuda.cudaMemcpy(dst.DevicePtr, src.DevicePtr, (nuint)src.SizeInBytes, (int)MemcpyKind.DeviceToDevice), "d2d");
            return;
        }
        Array.Copy(src.ToHost().Buffer, dst.ToHost().Buffer, src.ElementCount);
    }

    /// <summary>Asynchronous copy on a CUDA stream.</summary>
    public static void CopyToAsync(DeviceTensor<float> src, DeviceTensor<float> dst, CudaStreamHandle stream)
    {
        if (src.SizeInBytes != dst.SizeInBytes) throw new ArgumentException("Tensors differ in size.");

        if (src.IsOnDevice && dst.IsOnDevice)
        {
            NativeCheck.Ok(Cuda.cudaMemcpyAsync(dst.DevicePtr, src.DevicePtr, (nuint)src.SizeInBytes,
                                                (int)MemcpyKind.DeviceToDevice, stream.Handle), "d2d_async");
            return;
        }
        Array.Copy(src.ToHost().Buffer, dst.ToHost().Buffer, src.ElementCount);
    }

    /// <summary>Blocks the host until all queued work on <paramref name="stream"/> completes.</summary>
    public static void Synchronize(CudaStreamHandle stream) => CudaStreamManager.Shared.Synchronize(stream);

    // ── helpers ────────────────────────────────────────────────────────────────

    private static (int m, int k, int n) Dims(DeviceTensor<float> a, DeviceTensor<float> b)
    {
        var sa = a.Shape; var sb = b.Shape;
        if (sa.Length != 2 || sb.Length != 2) throw new ArgumentException("MatMul expects rank-2 tensors.");
        if (sa[1] != sb[0]) throw new ArgumentException($"Inner dimensions disagree: A is [{sa[0]},{sa[1]}], B is [{sb[0]},{sb[1]}].");
        return (sa[0], sa[1], sb[1]);
    }

    private static DeviceTensor<float> NewDeviceResult(int m, int n, int deviceId, CudaStreamHandle stream)
    {
        var c = DeviceTensor<float>.FromHost(new DenseTensor<float>(m, n)).To(Device.Cuda(deviceId));
        c.Stream = stream;
        return c;
    }
}
