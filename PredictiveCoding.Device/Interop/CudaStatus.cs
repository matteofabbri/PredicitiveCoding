namespace PredictiveCoding.Device.Interop;

/// <summary>
/// Status codes returned by the native bridge. 0 = success; the remaining values
/// mirror the categories the native layer reports (CUDA / cuBLAS / NCCL / alloc).
/// </summary>
public enum DeviceStatus
{
    Success          = 0,
    NotInitialized   = 1,
    NoDevice         = 2,
    OutOfMemory      = 3,
    InvalidValue     = 4,
    CudaError        = 5,
    CublasError      = 6,
    NcclError        = 7,
    NotSupported     = 8,
    Unknown          = 100,
}

/// <summary>Thrown when a native device operation fails.</summary>
public sealed class DeviceException : Exception
{
    public int Code { get; }

    public DeviceException(int code, string operation)
        : base($"Native device op '{operation}' failed with status {(DeviceStatus)code} ({code}).")
        => Code = code;
}

internal static class NativeCheck
{
    /// <summary>Throws a <see cref="DeviceException"/> when <paramref name="code"/> is non-zero.</summary>
    public static void Ok(int code, string operation)
    {
        if (code != 0) throw new DeviceException(code, operation);
    }
}
