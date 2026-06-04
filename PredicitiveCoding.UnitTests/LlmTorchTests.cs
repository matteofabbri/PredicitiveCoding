using PredictiveCoding.Device.Torch;

namespace PredicitiveCoding.UnitTests;

// The TorchSharp (Metal/MPS) backend. With no LibTorch native package present
// (as in CI) the probes must report unavailable WITHOUT throwing, and enabling a
// backend must be a no-op that leaves the existing GEMM provider in place. The
// torch provider is only constructed once a device is known to be available, so
// it is never instantiated here.
public class LlmTorchTests
{
    [Fact]
    public void Probes_AreFalse_WithoutLibTorch_AndDoNotThrow()
    {
        Assert.False(TorchBackend.MetalAvailable);
        Assert.False(TorchBackend.CudaAvailable);
    }

    [Fact]
    public void Enable_ReturnsFalse_WhenUnavailable()
    {
        // No LibTorch backend ⇒ enabling is a no-op (returns false, installs nothing).
        Assert.False(TorchBackend.EnableMetal());
        Assert.False(TorchBackend.EnableBest());
    }
}
