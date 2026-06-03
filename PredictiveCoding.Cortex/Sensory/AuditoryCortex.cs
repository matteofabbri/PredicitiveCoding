using PredictiveCoding.Core;
using PredictiveCoding.Expanded.Hierarchical;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// AuditoryCortex  —  hierarchical PC auditory pathway (A1 → belt → parabelt)
//
// Structurally identical to the visual cortex, but the input is a cochleagram /
// spectrogram [1, Freq, Time] and the topography is TONOTOPIC (frequency axis)
// rather than retinotopic.  A1 columns share weights across frequency–time, the
// belt/parabelt hierarchy abstracts spectro-temporal structure, and an optional
// readout classifies (e.g. phonemes).  Implemented by delegating to the generic
// VisualCortex with one input channel.
//
// References: Rao & Ballard (1999); Kumar et al. (2007) hierarchical auditory PC.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record AuditoryCortexConfig
{
    public int FrequencyBins { get; init; } = 32;   // tonotopic axis (height)
    public int TimeFrames    { get; init; } = 32;   // time axis (width)
    public int PatchSize     { get; init; } = 8;
    public int PatchStride   { get; init; } = 8;

    /// <summary>Shared A1 column. LayerSizes[0] must equal PatchSize².</summary>
    public NetworkConfig A1Column { get; init; } = new();

    /// <summary>Belt → parabelt hierarchy over the A1 feature map.</summary>
    public HierarchicalNetworkConfig BeltParabelt { get; init; } = new();

    /// <summary>Optional readout (e.g. phoneme / sound-class) on the top latent.</summary>
    public NetworkConfig? Classifier { get; init; }

    public string BasePath { get; init; } = "./cortex/auditory";

    internal VisualCortexConfig ToVisual() => new()
    {
        ImageChannels = 1,
        ImageHeight   = FrequencyBins,
        ImageWidth    = TimeFrames,
        PatchSize     = PatchSize,
        PatchStride   = PatchStride,
        V1Column      = A1Column,
        HigherAreas   = BeltParabelt,
        Classifier    = Classifier,
        BasePath      = BasePath,
    };
}

public sealed class AuditoryCortex : IDisposable
{
    private readonly VisualCortex _cortex;

    public AuditoryCortex(AuditoryCortexConfig config, Random rng)
        => _cortex = new VisualCortex(config.ToVisual(), rng);

    /// <summary>Dimensionality of the most abstract (parabelt) auditory latent.</summary>
    public int TopLatentDim => _cortex.ItLatentDim;

    /// <summary>Encodes a spectrogram [1,F,T] to the top auditory latent.</summary>
    public DenseTensor<float> Encode(DenseTensor<float> spectrogram, int steps)
        => _cortex.Encode(spectrogram, steps);

    /// <summary>Unsupervised learning of the auditory hierarchy on one spectrogram.</summary>
    public void Learn(DenseTensor<float> spectrogram, int steps)
        => _cortex.Learn(spectrogram, steps);

    /// <summary>Supervised learning (requires a Classifier in the config).</summary>
    public void LearnSupervised(DenseTensor<float> spectrogram, DenseTensor<float> label, int steps)
        => _cortex.LearnSupervised(spectrogram, label, steps);

    /// <summary>Classifies a spectrogram (requires a Classifier in the config).</summary>
    public int Recognize(DenseTensor<float> spectrogram, int steps)
        => _cortex.Predict(spectrogram, steps);

    /// <summary>Top-down reconstruction of the spectrogram.</summary>
    public DenseTensor<float> Reconstruct(DenseTensor<float> spectrogram, int steps)
        => _cortex.Reconstruct(spectrogram, steps);

    public void FlushAll() => _cortex.FlushAll();
    public void Dispose()  => _cortex.Dispose();
}
