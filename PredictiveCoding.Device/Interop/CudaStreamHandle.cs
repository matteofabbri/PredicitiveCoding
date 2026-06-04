namespace PredictiveCoding.Device.Interop;

/// <summary>
/// Opaque handle to a native CUDA stream (cudaStream_t). IntPtr.Zero is the
/// default/legacy stream.
/// </summary>
public readonly record struct CudaStreamHandle(IntPtr Handle)
{
    /// <summary>The default (legacy) CUDA stream.</summary>
    public static CudaStreamHandle Default => new(IntPtr.Zero);

    public bool IsDefault => Handle == IntPtr.Zero;

    public override string ToString() => IsDefault ? "stream:default" : $"stream:0x{Handle:X}";
}

/// <summary>Opaque handle to a native CUDA event (cudaEvent_t).</summary>
public readonly record struct CudaEventHandle(IntPtr Handle)
{
    public bool IsNull => Handle == IntPtr.Zero;
}

/// <summary>Opaque handle to a native NCCL communicator (ncclComm_t).</summary>
public readonly record struct NcclCommHandle(IntPtr Handle)
{
    public bool IsNull => Handle == IntPtr.Zero;
}
