using PredictiveCoding.Core;
using PredictiveCoding.Expanded.MultiModal;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// PosteriorParietalCortex  —  sensorimotor coordinate transforms / intentional maps
//
// The posterior parietal cortex transforms sensory signals between reference frames
// (e.g. retinal coordinates + eye position → a body-centred target) and forms the
// "intention" that bridges perception to action — the dorsal-stream → motor link.
// Implemented over a MultiModalNetwork that binds the input coordinate frames into a
// common latent and reads out the target zone / intention.
//
// References: Andersen & Buneo (2002) intentional maps in PPC; Pouget & Snyder (2000).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record PosteriorParietalCortexConfig
{
    /// <summary>Multimodal integrator (e.g. retinal coords + eye position → common frame).</summary>
    public MultiModalNetworkConfig MultiModal { get; init; } = new();

    public string BasePath { get; init; } = "./cortex/parietal";

    public void Validate() => MultiModal.Validate();
}

public sealed class PosteriorParietalCortex : IDisposable
{
    private readonly MultiModalNetwork _net;

    public int ModalityCount => _net.ModalityCount;

    public PosteriorParietalCortex(PosteriorParietalCortexConfig config, Random rng)
    {
        config.Validate();
        _net = new MultiModalNetwork(config.MultiModal with { BasePath = config.BasePath }, rng);
    }

    /// <summary>Transforms the input coordinate frames into the common (e.g. body-centred) latent.</summary>
    public DenseTensor<float> Transform(IReadOnlyDictionary<int, DenseTensor<float>> frames, int steps)
        => _net.EncodeAll(frames, steps);

    /// <summary>Learns the transform: (coordinate frames) → target-zone / intention label.</summary>
    public void Learn(IReadOnlyDictionary<int, DenseTensor<float>> frames, DenseTensor<float> targetLabel, int steps)
        => _net.Learn(frames, targetLabel, steps);

    /// <summary>Locates the target zone / intention from the (possibly partial) coordinate frames.</summary>
    public int Locate(IReadOnlyDictionary<int, DenseTensor<float>> frames, int steps)
        => _net.Predict(frames, steps);

    public void FlushAll() => _net.FlushAll();
    public void Dispose()  => _net.Dispose();
}
