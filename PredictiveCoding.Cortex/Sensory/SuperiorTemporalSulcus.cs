using PredictiveCoding.Core;
using PredictiveCoding.Networks.MultiModal;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// SuperiorTemporalSulcus (STS)  —  multimodal social perception
//
// The STS integrates *changeable* social signals — biological motion, gaze, facial
// expression, and voice — binding what is seen and heard about an agent into a single
// social percept (e.g. pairing a face with its voice, reading an action's intent).
// Implemented over a `MultiModalNetwork` that fuses the visual (biological-motion) and
// auditory (voice) streams into a common social latent and reads out the agent /
// social category, mirroring `PosteriorParietalCortex`'s multimodal binding.
//
// References: Allison, Puce & McCarthy (2000) social perception in STS;
//             Beauchamp (2005) audiovisual integration in STS.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record SuperiorTemporalSulcusConfig
{
    /// <summary>Multimodal integrator (e.g. biological motion + voice → social latent).</summary>
    public MultiModalNetworkConfig MultiModal { get; init; } = new();

    public string BasePath { get; init; } = "./cortex/sts";

    public void Validate() => MultiModal.Validate();
}

public sealed class SuperiorTemporalSulcus : IDisposable
{
    private readonly MultiModalNetwork _net;

    public int ModalityCount => _net.ModalityCount;

    public SuperiorTemporalSulcus(SuperiorTemporalSulcusConfig config, Random rng)
    {
        config.Validate();
        _net = new MultiModalNetwork(config.MultiModal with { BasePath = config.BasePath }, rng);
    }

    /// <summary>Binds the social streams (e.g. motion + voice) into the common social latent.</summary>
    public DenseTensor<float> Integrate(IReadOnlyDictionary<int, DenseTensor<float>> streams, int steps)
        => _net.EncodeAll(streams, steps);

    /// <summary>Learns the (streams) → social-category / agent label association.</summary>
    public void Learn(IReadOnlyDictionary<int, DenseTensor<float>> streams, DenseTensor<float> socialLabel, int steps)
        => _net.Learn(streams, socialLabel, steps);

    /// <summary>Reads out the social category / agent from the (possibly partial) streams
    /// — supports cross-modal completion (recognise the agent from voice alone, etc.).</summary>
    public int Classify(IReadOnlyDictionary<int, DenseTensor<float>> streams, int steps)
        => _net.Predict(streams, steps);

    public void FlushAll() => _net.FlushAll();
    public void Dispose() => _net.Dispose();
}
