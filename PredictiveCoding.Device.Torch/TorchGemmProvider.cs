using PredictiveCoding.LLM.Infrastructure;
using TorchSharp;

namespace PredictiveCoding.Device.Torch;

// ─────────────────────────────────────────────────────────────────────────────
// TorchGemmProvider  —  GEMM backend on TorchSharp (LibTorch), for Apple Metal.
//
// LibTorch's MPS backend runs on Apple-Silicon GPUs (Metal), so this is the
// managed way to get GPU matmul on Metal — there is no flat cudart/cublas-style
// C ABI for Metal. The same provider also drives CUDA and CPU torch devices.
//
// It plugs into the LLM stack via IGemmProvider exactly like CudaGemmProvider, so
// the two coexist: the direct-cuBLAS path stays the default on NVIDIA, this one
// covers Metal (and optionally CUDA/CPU through torch). Any failure (missing
// LibTorch backend, unsupported op) falls back to the CPU provider.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class TorchGemmProvider : IGemmProvider
{
    private readonly torch.Device _device;
    private readonly IGemmProvider _cpuFallback = new CpuGemmProvider();

    public TorchGemmProvider(torch.Device device) => _device = device;

    public static TorchGemmProvider OnMetal() => new(torch.device("mps"));
    public static TorchGemmProvider OnCuda()  => new(torch.device("cuda"));
    public static TorchGemmProvider OnCpu()   => new(torch.device("cpu"));

    /// <summary>The torch device type, e.g. "MPS", "CUDA", "CPU".</summary>
    public string DeviceType => _device.type.ToString();

    public float[] MatMul(float[] a, int m, int k, float[] b, int n)
    {
        try
        {
            using var ta = torch.from_array(a).reshape(m, k).to(_device);
            using var tb = torch.from_array(b).reshape(k, n).to(_device);
            using var tc = torch.matmul(ta, tb).cpu();
            return tc.data<float>().ToArray();
        }
        catch
        {
            return _cpuFallback.MatMul(a, m, k, b, n);   // LibTorch backend missing / op unsupported
        }
    }
}
