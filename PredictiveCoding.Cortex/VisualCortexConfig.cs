using PredictiveCoding.Networks.Hierarchical;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// VisualCortexConfig  —  configuration for a hierarchical predictive-coding
//                        visual cortex (V1 → V2 → V4 → IT).
//
// The model mirrors the ventral visual stream:
//
//   • V1  : one weight-shared cortical column (a PCN) applied to every P×P
//           receptive field of the image.  Learns local edge / Gabor-like
//           features (Rao & Ballard 1999).  V1Column.LayerSizes[0] = C·P·P,
//           V1Column.LayerSizes[^1] = number of V1 feature maps K.
//   • V2…IT: a deep modular hierarchy (HierarchicalNetwork) over the flattened
//           V1 feature map, building progressively abstract, larger-scale
//           representations.  HigherAreas.Modules[0].LayerSizes[0] must equal the
//           flattened V1 feature-map size  GridH·GridW·K.
//   • Readout (optional): a PCN classifier on the IT latent for recognition.
//           Classifier.LayerSizes[0] = IT latent dim, LayerSizes[^1] = #classes.
//
// References:
//   Rao & Ballard (1999) — predictive coding of natural images / V1.
//   Felleman & Van Essen (1991) — hierarchical organisation of visual cortex.
//   Friston (2005) — hierarchical free-energy minimisation.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record VisualCortexConfig
{
    // ── Retina / image geometry ────────────────────────────────────────────────

    public int ImageChannels { get; init; } = 3;
    public int ImageHeight   { get; init; } = 32;
    public int ImageWidth    { get; init; } = 32;

    /// <summary>Receptive-field size P (each V1 column sees a P×P patch).</summary>
    public int PatchSize { get; init; } = 8;

    /// <summary>Stride between neighbouring receptive fields (overlap = P − stride).</summary>
    public int PatchStride { get; init; } = 4;

    // ── Cortical areas ──────────────────────────────────────────────────────────

    /// <summary>
    /// Shared V1 cortical-column PCN.  Input = C·P·P, top = K (V1 feature maps).
    /// </summary>
    public NetworkConfig V1Column { get; init; } = new();

    /// <summary>Deep modular hierarchy implementing V2 → V4 → IT.</summary>
    public HierarchicalNetworkConfig HigherAreas { get; init; } = new();

    /// <summary>
    /// Optional readout classifier on the IT latent.  Null ⇒ the cortex is purely
    /// unsupervised (encode / reconstruct only).
    /// </summary>
    public NetworkConfig? Classifier { get; init; }

    public string BasePath { get; init; } = "./cortex";

    // ── Derived geometry ────────────────────────────────────────────────────────

    public int GridHeight    => RetinotopicMap.GridSize(ImageHeight, PatchSize, PatchStride);
    public int GridWidth     => RetinotopicMap.GridSize(ImageWidth,  PatchSize, PatchStride);
    public int V1FeatureDim  => V1Column.LayerSizes[^1];
    public int FeatureMapDim => GridHeight * GridWidth * V1FeatureDim;
    public int ItLatentDim   => HigherAreas.Modules[^1].LayerSizes[^1];

    public void Validate()
    {
        if (ImageChannels < 1 || ImageHeight < 1 || ImageWidth < 1)
            throw new InvalidOperationException("Image dimensions must be >= 1.");
        if (PatchSize < 1 || PatchSize > ImageHeight || PatchSize > ImageWidth)
            throw new InvalidOperationException("PatchSize must be in [1, min(Height, Width)].");
        if (PatchStride < 1)
            throw new InvalidOperationException("PatchStride must be >= 1.");

        int expectedColIn = ImageChannels * PatchSize * PatchSize;
        if (V1Column.LayerSizes[0] != expectedColIn)
            throw new InvalidOperationException(
                $"V1Column.LayerSizes[0] ({V1Column.LayerSizes[0]}) must equal C·P·P ({expectedColIn}).");

        HigherAreas.Validate();
        if (HigherAreas.Modules[0].LayerSizes[0] != FeatureMapDim)
            throw new InvalidOperationException(
                $"HigherAreas.Modules[0].LayerSizes[0] ({HigherAreas.Modules[0].LayerSizes[0]}) must equal " +
                $"the flattened V1 feature map GridH·GridW·K ({FeatureMapDim}).");

        if (Classifier is not null && Classifier.LayerSizes[0] != ItLatentDim)
            throw new InvalidOperationException(
                $"Classifier.LayerSizes[0] ({Classifier.LayerSizes[0]}) must equal the IT latent dim ({ItLatentDim}).");
    }
}
