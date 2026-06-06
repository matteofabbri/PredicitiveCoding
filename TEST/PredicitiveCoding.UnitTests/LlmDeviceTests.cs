using PredictiveCoding.Core;
using PredictiveCoding.Device;
using PredictiveCoding.Device.Interop;
using PredictiveCoding.Device.Nccl;

namespace PredicitiveCoding.UnitTests;

// Exercises the device-management layer on its CPU-fallback path (no native lib /
// no GPU in CI). The same API uses CUDA when DeviceRuntime.CudaAvailable is true.
public class LlmDeviceTests
{
    [Fact]
    public void Device_FactoriesAndToString()
    {
        Assert.True(Device.Cpu.IsCpu);
        Assert.Equal(1, Device.Cuda(1).Id);
        Assert.True(Device.Cuda(1).IsCuda);
        Assert.Equal("cpu", Device.Cpu.ToString());
        Assert.Equal("cuda:2", Device.Cuda(2).ToString());
    }

    [Fact]
    public void Runtime_FallsBackToCpu_WhenNoCuda()
    {
        if (DeviceRuntime.CudaAvailable) return;          // skip on a real GPU box
        Assert.False(DeviceRuntime.CudaAvailable);
        Assert.Equal(Device.Cpu, DeviceRuntime.Resolve(Device.Cuda()));
        Assert.Equal(Device.Cpu, DeviceRuntime.Resolve(Device.CudaMulti));
    }

    [Fact]
    public void DeviceTensor_RoundTripsThroughTo()
    {
        var host = new DenseTensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
        using var moved = host.ToDevice(Device.Cuda());   // CPU fallback here
        var back = moved.ToHostTensor();
        Assert.Equal(host.Buffer, back.Buffer);
        Assert.Equal(new[] { 2, 3 }, back.Dimensions);
    }

    [Fact]
    public void TensorOps_MatMul_ComputesProduct()
    {
        var a = new DenseTensor<float>(new float[] { 1, 2, 3, 4, 5, 6 }, 2, 3).ToDevice(Device.Cuda());
        var b = new DenseTensor<float>(new float[] { 7, 8, 9, 10, 11, 12 }, 3, 2).ToDevice(Device.Cuda());
        using var c = TensorOps.MatMul(a, b);
        var r = c.ToHostTensor();

        // [1 2 3;4 5 6] · [7 8;9 10;11 12] = [58 64; 139 154]
        Assert.Equal(new[] { 2, 2 }, r.Dimensions);
        Assert.Equal(58f, r.Buffer[0], 3);
        Assert.Equal(64f, r.Buffer[1], 3);
        Assert.Equal(139f, r.Buffer[2], 3);
        Assert.Equal(154f, r.Buffer[3], 3);
    }

    [Fact]
    public void CudaGemmProvider_FallsBackToCpuResult()
    {
        var rng = new Random(1);
        int m = 4, k = 5, n = 3;
        var a = new float[m * k]; var b = new float[k * n];
        for (int i = 0; i < a.Length; i++) a[i] = (float)rng.NextDouble();
        for (int i = 0; i < b.Length; i++) b[i] = (float)rng.NextDouble();

        var cuda = new CudaGemmProvider().MatMul(a, m, k, b, n);
        var cpu  = new PredictiveCoding.LLM.Infrastructure.CpuGemmProvider().MatMul(a, m, k, b, n);
        for (int i = 0; i < cuda.Length; i++) Assert.Equal(cpu[i], cuda[i], 4);
    }

    [Fact]
    public void Nccl_CpuMock_Collectives()
    {
        using var nccl = new NcclProvider([0, 1, 2]);     // 3 ranks; CPU mock (no multi-GPU)
        Assert.Equal(3, nccl.RankCount);

        var allreduce = new[] { new[] { 1f, 2f, 3f }, new[] { 10f, 20f, 30f }, new[] { 100f, 200f, 300f } };
        nccl.AllReduce(allreduce);
        Assert.All(allreduce, b => Assert.Equal(new[] { 111f, 222f, 333f }, b));

        var bcast = new[] { new[] { 0f, 0f }, new[] { 9f, 8f }, new[] { 0f, 0f } };
        nccl.Broadcast(bcast, root: 1);
        Assert.All(bcast, b => Assert.Equal(new[] { 9f, 8f }, b));

        var gatherSend = new[] { new[] { 1f }, new[] { 2f }, new[] { 3f } };
        var gatherRecv = new[] { new float[3], new float[3], new float[3] };
        nccl.AllGather(gatherSend, gatherRecv);
        Assert.All(gatherRecv, b => Assert.Equal(new[] { 1f, 2f, 3f }, b));
    }

    [Fact]
    public void Registry_AutoSelect_InstallsWorkingProvider()
    {
        var device = DeviceRegistry.AutoSelect();
        Assert.Equal(DeviceRuntime.CudaAvailable ? DeviceKind.Cuda : DeviceKind.Cpu, device.Kind);

        // The installed LLM GEMM provider still computes correctly.
        var c = PredictiveCoding.LLM.Infrastructure.Gemm.MatMul([1, 2, 3, 4], 2, 2, [1, 0, 0, 1], 2);
        Assert.Equal(new[] { 1f, 2f, 3f, 4f }, c);
        DeviceRegistry.UseCpu();
    }
}
