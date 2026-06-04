namespace PredictiveCoding.Device;

// ─────────────────────────────────────────────────────────────────────────────
// Device  —  a logical compute device (CPU / CUDA / Metal / ROCm).
//
// PyTorch-style handle. The backend that actually executes work is selected by
// DeviceRuntime / DeviceRegistry; if the requested accelerator is unavailable the
// system transparently falls back to the CPU.
// ─────────────────────────────────────────────────────────────────────────────

public enum DeviceKind
{
    Cpu,
    Cuda,
    CudaMulti,
    Metal,
    Rocm,
}

public readonly record struct Device(DeviceKind Kind, int Id = 0)
{
    /// <summary>The host CPU device.</summary>
    public static Device Cpu => new(DeviceKind.Cpu);

    /// <summary>A single CUDA device (default ordinal 0). Use <c>Device.Cuda(id)</c> to pick one.</summary>
    public static Device Cuda(int id = 0) => new(DeviceKind.Cuda, id);

    /// <summary>A logical "all CUDA devices" handle for collective/tensor-parallel ops.</summary>
    public static Device CudaMulti => new(DeviceKind.CudaMulti);

    public static Device Metal(int id = 0) => new(DeviceKind.Metal, id);
    public static Device Rocm(int id = 0)  => new(DeviceKind.Rocm, id);

    public bool IsCpu => Kind == DeviceKind.Cpu;
    public bool IsGpu => Kind is DeviceKind.Cuda or DeviceKind.CudaMulti or DeviceKind.Metal or DeviceKind.Rocm;
    public bool IsCuda => Kind is DeviceKind.Cuda or DeviceKind.CudaMulti;

    public override string ToString() => Kind == DeviceKind.Cpu ? "cpu" : $"{Kind.ToString().ToLowerInvariant()}:{Id}";
}
