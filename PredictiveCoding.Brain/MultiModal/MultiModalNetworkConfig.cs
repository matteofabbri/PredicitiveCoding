using PredictiveCoding.Networks;

namespace PredictiveCoding.Brain.MultiModal;

// ─────────────────────────────────────────────────────────────────────────────
// MultiModalNetworkConfig  —  configuration for a multi-sensory PCN
//
// Topology:
//
//   modality_0 (vision):   x_0 [d_0] → PCN_0 → z_0 [l_0]  ─┐
//   modality_1 (audio):    x_1 [d_1] → PCN_1 → z_1 [l_1]  ─├─ [z_0|z_1|…] → associative PCN → label
//   modality_2 (text):     x_2 [d_2] → PCN_2 → z_2 [l_2]  ─┘
//
// The associative network's input size must equal Σ_i l_i.
// Missing modalities at inference time are zero-padded.
//
// Cross-modal generation (optional):
//   Requires TrainAssociativeGeneratively = true.
//   The associative network also learns to reconstruct [z_0|z_1|…] from
//   its top-layer representation, enabling:
//     source_modality → z_assoc_top → Generate → z_full_hat → target z_i → target x_i
// ─────────────────────────────────────────────────────────────────────────────

public sealed record MultiModalNetworkConfig
{
    /// <summary>
    /// One entry per sensory modality, ordered consistently with the index
    /// used in <see cref="MultiModalNetwork"/> API calls.
    /// </summary>
    public ModalityConfig[] Modalities { get; init; } = [];

    /// <summary>
    /// Associative integration network.
    /// LayerSizes[0] must equal the sum of all modality latent sizes.
    /// </summary>
    public NetworkConfig AssociativeConfig { get; init; } = new();

    /// <summary>
    /// When true, the associative network is also trained generatively:
    /// it learns to reconstruct the full concatenated latent vector from its
    /// top-layer representation.  Required for <see cref="MultiModalNetwork.CrossModalGenerate"/>.
    /// </summary>
    public bool TrainAssociativeGeneratively { get; init; } = false;

    public string BasePath { get; init; } = "./brain/multimodal";

    public void Validate()
    {
        if (Modalities.Length < 2)
            throw new InvalidOperationException("MultiModalNetwork requires at least 2 modalities.");

        int expectedInput = Modalities.Sum(m => m.NetworkConfig.LayerSizes[^1]);
        if (AssociativeConfig.LayerSizes[0] != expectedInput)
            throw new InvalidOperationException(
                $"AssociativeConfig.LayerSizes[0] ({AssociativeConfig.LayerSizes[0]}) must equal " +
                $"the sum of all modality latent sizes ({expectedInput}). " +
                $"Modality latent sizes: [{string.Join(", ", Modalities.Select(m => m.NetworkConfig.LayerSizes[^1]))}].");
    }
}
