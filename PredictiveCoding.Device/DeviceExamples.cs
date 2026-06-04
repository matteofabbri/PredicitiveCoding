using PredictiveCoding.Core;
using PredictiveCoding.Device.Interop;
using PredictiveCoding.Device.Nccl;
using PredictiveCoding.Device.Streams;

namespace PredictiveCoding.Device;

// ─────────────────────────────────────────────────────────────────────────────
// DeviceExamples  —  runnable usage samples (work on CPU; use GPU when present).
// ─────────────────────────────────────────────────────────────────────────────

public static class DeviceExamples
{
    /// <summary>1. Create a tensor on the CPU.</summary>
    public static DeviceTensor<float> CreateCpuTensor()
    {
        var host = new DenseTensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
        return DeviceTensor<float>.FromHost(host);   // == host.AsDeviceTensor()
    }

    /// <summary>2. Move a tensor onto the GPU (falls back to CPU automatically).</summary>
    public static DeviceTensor<float> MoveToGpu()
    {
        var host = new DenseTensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
        return host.ToDevice(Device.Cuda());         // Device.Cuda(0); CUDA→CPU fallback if unavailable
    }

    /// <summary>3. Matrix multiply on the GPU (cuBLASLt) — A[2,3] · B[3,2] = C[2,2].</summary>
    public static DenseTensor<float> MatMulGpu()
    {
        var a = new DenseTensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3).ToDevice(Device.Cuda());
        var b = new DenseTensor<float>(new float[] { 7, 8, 9, 10, 11, 12 }, 3, 2).ToDevice(Device.Cuda());
        using var c = TensorOps.MatMul(a, b);
        a.Dispose(); b.Dispose();
        return c.ToHostTensor();                     // 6. back to the CPU
    }

    /// <summary>4. Multi-GPU: reduce a per-device buffer across all devices (NCCL or CPU mock).</summary>
    public static float[] MatMulMultiGpu()
    {
        using var nccl = NcclProvider.ForAllDevices();          // ranks = #CUDA devices (1 on CPU)
        var perRank = Enumerable.Range(0, nccl.RankCount)
                                .Select(r => new float[] { 1f * (r + 1), 2f * (r + 1) })
                                .ToArray();
        nccl.AllReduce(perRank, ReduceOp.Sum);                  // each rank now holds the global sum
        return perRank[0];
    }

    /// <summary>5. Asynchronous matmul on a dedicated CUDA stream, then synchronize.</summary>
    public static DenseTensor<float> AsyncWithStreams()
    {
        var a = new DenseTensor<float>(new float[] { 1, 0, 0, 1 }, 2, 2).ToDevice(Device.Cuda());
        var b = new DenseTensor<float>(new float[] { 5, 6, 7, 8 }, 2, 2).ToDevice(Device.Cuda());

        CudaStreamHandle stream = CudaStreamManager.Shared.CreateStream(deviceId: 0);
        using var c = TensorOps.MatMulAsync(a, b, stream);
        TensorOps.Synchronize(stream);               // 10. wait for the queued work

        a.Dispose(); b.Dispose();
        return c.ToHostTensor();
    }

    /// <summary>Installs the CUDA GEMM backend into the LLM stack when a GPU is present.</summary>
    public static Device SelectBestBackend() => DeviceRegistry.AutoSelect();
}
