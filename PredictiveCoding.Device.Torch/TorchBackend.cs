using PredictiveCoding.LLM.Infrastructure;
using TorchSharp;

namespace PredictiveCoding.Device.Torch;

// ─────────────────────────────────────────────────────────────────────────────
// TorchBackend  —  detects torch devices and installs the TorchSharp GEMM backend.
//
// Availability is probed by trying to place a tiny tensor on the device; if the
// LibTorch backend isn't present the probe returns false and nothing is installed
// (the existing provider — CPU, or the direct-cuBLAS CudaGemmProvider — stays).
// ─────────────────────────────────────────────────────────────────────────────

public static class TorchBackend
{
    /// <summary>True when LibTorch can place a tensor on the Apple Metal (MPS) device.</summary>
    public static bool MetalAvailable => Probe("mps");

    /// <summary>True when LibTorch can use a CUDA device.</summary>
    public static bool CudaAvailable => Probe("cuda");

    /// <summary>Installs the Metal (MPS) backend into the LLM GEMM stack. Returns false if unavailable.</summary>
    public static bool EnableMetal()
    {
        if (!MetalAvailable) return false;
        Gemm.Provider = new TorchGemmProvider(torch.device("mps"));
        return true;
    }

    /// <summary>Installs the CUDA backend via torch (alternative to the direct-cuBLAS provider).</summary>
    public static bool EnableCuda()
    {
        if (!CudaAvailable) return false;
        Gemm.Provider = new TorchGemmProvider(torch.device("cuda"));
        return true;
    }

    /// <summary>Installs the best torch device (Metal → CUDA), or leaves the current provider.</summary>
    public static bool EnableBest() => EnableMetal() || EnableCuda();

    private static bool Probe(string device)
    {
        try
        {
            using var t = torch.zeros(new long[] { 1 }, device: torch.device(device));
            return true;
        }
        catch
        {
            return false;   // LibTorch / backend / device not available
        }
    }
}
