using PredictiveCoding.Core;
using PredictiveCoding.Networks.MultiModal;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// AssociationCortex  —  multimodal binding / heteromodal integration
//
// Higher-order association cortex (parietal / superior-temporal) binds inputs
// from several sensory streams into a common representation and supports
// cross-modal completion (e.g. infer the sound of a seen object).  Implemented
// as a specialisation of the MultiModalNetwork: per-modality encoders feed a
// shared associative PCN.
//
// References: Mesulam (1998) heteromodal cortex; Denk et al. (2023) multisensory PC.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record AssociationCortexConfig
{
    public MultiModalNetworkConfig MultiModal { get; init; } = new();
    public string BasePath { get; init; } = "./cortex/association";

    public void Validate() => MultiModal.Validate();
}

public sealed class AssociationCortex : IDisposable
{
    private readonly MultiModalNetwork _net;

    public int ModalityCount => _net.ModalityCount;

    public AssociationCortex(AssociationCortexConfig config, Random rng)
    {
        config.Validate();
        _net = new MultiModalNetwork(config.MultiModal with { BasePath = config.BasePath }, rng);
    }

    /// <summary>Encodes and concatenates the present modalities into the joint latent.</summary>
    public DenseTensor<float> Bind(IReadOnlyDictionary<int, DenseTensor<float>> modalInputs, int steps)
        => _net.EncodeAll(modalInputs, steps);

    /// <summary>Supervised learning over a (possibly partial) set of modalities.</summary>
    public void Learn(IReadOnlyDictionary<int, DenseTensor<float>> modalInputs, DenseTensor<float> label, int steps)
        => _net.Learn(modalInputs, label, steps);

    /// <summary>Classifies from any non-empty subset of modalities (missing → zero).</summary>
    public int Predict(IReadOnlyDictionary<int, DenseTensor<float>> modalInputs, int steps)
        => _net.Predict(modalInputs, steps);

    /// <summary>Generates the target modality from the source modality alone.</summary>
    public DenseTensor<float> CrossModal(int sourceIndex, DenseTensor<float> sourceInput, int targetIndex, int steps)
        => _net.CrossModalGenerate(sourceIndex, sourceInput, targetIndex, steps);

    public void FlushAll() => _net.FlushAll();
    public void Dispose()  => _net.Dispose();
}
