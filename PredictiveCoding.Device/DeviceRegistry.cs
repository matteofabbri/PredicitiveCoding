using PredictiveCoding.LLM.Infrastructure;

namespace PredictiveCoding.Device;

// ─────────────────────────────────────────────────────────────────────────────
// DeviceRegistry  —  selects and installs the active compute backend.
//
// Installs a CUDA GEMM provider into the LLM stack (replacing the CPU
// IGemmProvider) when CUDA is available, so every existing matmul in the model
// runs on the GPU with no code changes. Everything falls back to the CPU
// automatically.
// ─────────────────────────────────────────────────────────────────────────────

public static class DeviceRegistry
{
    /// <summary>True when the CUDA GEMM backend is currently installed.</summary>
    public static bool CudaEnabled { get; private set; }

    /// <summary>The device the system will use by default.</summary>
    public static Device DefaultDevice => DeviceRuntime.CudaAvailable ? Device.Cuda(0) : Device.Cpu;

    /// <summary>
    /// Installs the CUDA GEMM backend (the provider itself falls back to CPU per call
    /// if CUDA is missing). Returns true if a real CUDA device is present.
    /// </summary>
    public static bool EnableCuda(int deviceId = 0)
    {
        Gemm.Provider = new CudaGemmProvider(deviceId);
        CudaEnabled = DeviceRuntime.CudaAvailable;
        return CudaEnabled;
    }

    /// <summary>Restores the pure-CPU GEMM backend.</summary>
    public static void UseCpu()
    {
        Gemm.Provider = new CpuGemmProvider();
        CudaEnabled = false;
    }

    /// <summary>Picks CUDA when available, CPU otherwise.</summary>
    public static Device AutoSelect()
    {
        if (DeviceRuntime.CudaAvailable) EnableCuda();
        else UseCpu();
        return DefaultDevice;
    }
}
