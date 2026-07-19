using PredictiveCoding.Networks;

namespace PredictiveCoding.Networks.MultiModal;

/// <summary>
/// Configuration for a single sensory modality within a <see cref="MultiModalNetwork"/>.
/// </summary>
public sealed record ModalityConfig
{
    /// <summary>Human-readable name, e.g. "vision", "audio", "text".</summary>
    public string Name { get; init; } = "modality";

    /// <summary>
    /// PC network for this modality.
    /// LayerSizes[0]  = raw input dimensionality for this modality.
    /// LayerSizes[^1] = latent representation size contributed to the joint space.
    /// </summary>
    public NetworkConfig NetworkConfig { get; init; } = new();
}
