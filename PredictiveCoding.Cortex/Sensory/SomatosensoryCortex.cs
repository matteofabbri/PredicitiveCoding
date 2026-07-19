using PredictiveCoding.Core;
using PredictiveCoding.Networks.Hierarchical;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// SomatosensoryCortex  —  hierarchical PC touch pathway (S1 → S2 → …)
//
// A SOMATOTOPIC version of the same cortical microcircuit: the input is a "skin
// map" [R, H, W] where R is the number of receptor types (pressure, vibration,
// temperature, …) and H×W tiles the body surface.  S1 columns share weights
// across body locations; a higher hierarchy abstracts tactile structure.
// Delegates to the generic VisualCortex with R input channels.
//
// References: Rao & Ballard (1999); Adams, Shipp & Friston (2013) PC of action.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record SomatosensoryCortexConfig
{
    public int ReceptorChannels { get; init; } = 2;   // e.g. pressure + vibration
    public int MapHeight        { get; init; } = 24;
    public int MapWidth         { get; init; } = 24;
    public int PatchSize        { get; init; } = 8;
    public int PatchStride      { get; init; } = 8;

    /// <summary>Shared S1 column. LayerSizes[0] must equal ReceptorChannels·PatchSize².</summary>
    public NetworkConfig S1Column { get; init; } = new();

    /// <summary>Higher somatosensory areas over the S1 feature map.</summary>
    public HierarchicalNetworkConfig HigherAreas { get; init; } = new();

    /// <summary>Optional readout (e.g. texture / object-in-hand) on the top latent.</summary>
    public NetworkConfig? Classifier { get; init; }

    public string BasePath { get; init; } = "./cortex/somatosensory";

    internal VisualCortexConfig ToVisual() => new()
    {
        ImageChannels = ReceptorChannels,
        ImageHeight   = MapHeight,
        ImageWidth    = MapWidth,
        PatchSize     = PatchSize,
        PatchStride   = PatchStride,
        V1Column      = S1Column,
        HigherAreas   = HigherAreas,
        Classifier    = Classifier,
        BasePath      = BasePath,
    };
}

public sealed class SomatosensoryCortex : IDisposable
{
    private readonly VisualCortex _cortex;

    public SomatosensoryCortex(SomatosensoryCortexConfig config, Random rng)
        => _cortex = new VisualCortex(config.ToVisual(), rng);

    public int TopLatentDim => _cortex.ItLatentDim;

    public DenseTensor<float> Encode(DenseTensor<float> skinMap, int steps)
        => _cortex.Encode(skinMap, steps);

    public void Learn(DenseTensor<float> skinMap, int steps)
        => _cortex.Learn(skinMap, steps);

    public void LearnSupervised(DenseTensor<float> skinMap, DenseTensor<float> label, int steps)
        => _cortex.LearnSupervised(skinMap, label, steps);

    public int Recognize(DenseTensor<float> skinMap, int steps)
        => _cortex.Predict(skinMap, steps);

    public DenseTensor<float> Reconstruct(DenseTensor<float> skinMap, int steps)
        => _cortex.Reconstruct(skinMap, steps);

    public void FlushAll() => _cortex.FlushAll();
    public void Dispose()  => _cortex.Dispose();
}
