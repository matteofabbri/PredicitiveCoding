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
        RocmEnabled = false;
        return CudaEnabled;
    }

    /// <summary>True when the ROCm GEMM backend is currently installed (A4 mirror).</summary>
    public static bool RocmEnabled { get; private set; }

    /// <summary>
    /// Installs the ROCm (hipBLAS) GEMM backend — the A4 mirror of
    /// <see cref="EnableCuda"/>, UNTESTED-WITHOUT-HARDWARE (the provider falls
    /// back to CPU per call when no ROCm runtime is present). Returns true if a
    /// real ROCm device is present.
    /// </summary>
    public static bool EnableRocm(int deviceId = 0)
    {
        Gemm.Provider = new RocmGemmProvider(deviceId);
        CudaEnabled = false;
        RocmEnabled = DeviceRuntime.RocmAvailable;
        return RocmEnabled;
    }

    /// <summary>Restores the pure-CPU GEMM backend.</summary>
    public static void UseCpu()
    {
        Gemm.Provider = new CpuGemmProvider();
        CudaEnabled = false;
        RocmEnabled = false;
    }

    /// <summary>Picks CUDA when available, then ROCm, then CPU.</summary>
    public static Device AutoSelect()
    {
        if (DeviceRuntime.CudaAvailable) EnableCuda();
        else if (DeviceRuntime.RocmAvailable) EnableRocm();
        else UseCpu();
        return DefaultDevice;
    }
}
