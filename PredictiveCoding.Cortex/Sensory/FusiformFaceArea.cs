using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// FusiformFaceArea (FFA)  —  face-selective ventral-stream module
//
// A category-selective region of the ventral stream that responds preferentially to
// faces and supports face individuation. Sitting downstream of `VisualCortex` (IT), it
// maps a high-level visual latent to a compact face-identity embedding, learned from
// examples — the specialised "head" that turns generic object features into a face
// code. Same PCN recipe as `Amygdala`/`OrbitofrontalCortex`, just a different target.
//
// References: Kanwisher, McDermott & Chun (1997) FFA; Tsao & Livingstone (2008)
//             mechanisms of face processing.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record FusiformFaceAreaConfig
{
    /// <summary>Recognition PCN: visual latent → face-identity embedding.
    /// LayerSizes[^1] must equal <see cref="FaceLatentDim"/>.</summary>
    public NetworkConfig Recognition { get; init; } = new();

    /// <summary>Dimensionality of the face-identity embedding.</summary>
    public int FaceLatentDim { get; init; } = 16;

    public string BasePath { get; init; } = "./cortex/ffa";

    public void Validate()
    {
        if (FaceLatentDim < 1)
            throw new InvalidOperationException("FaceLatentDim must be >= 1.");
        if (Recognition.LayerSizes[^1] != FaceLatentDim)
            throw new InvalidOperationException(
                $"Recognition.LayerSizes[^1] ({Recognition.LayerSizes[^1]}) must equal FaceLatentDim ({FaceLatentDim}).");
    }
}

public sealed class FusiformFaceArea : IDisposable
{
    private readonly Network _net;
    private readonly FusiformFaceAreaConfig _config;

    public int VisualDim => _net.Configuration.LayerSizes[0];
    public int FaceLatentDim => _config.FaceLatentDim;

    public FusiformFaceArea(FusiformFaceAreaConfig config, Random rng)
    {
        config.Validate();
        _config = config;
        _net = new Network(config.Recognition with { Path = config.BasePath }, rng);
    }

    /// <summary>Encodes a visual (IT) latent into a face-identity embedding.</summary>
    public DenseTensor<float> Encode(DenseTensor<float> visualLatent, int steps)
        => _net.InferGetTop(visualLatent, steps);

    /// <summary>Learns the visual-latent → face-identity association.</summary>
    public void Learn(DenseTensor<float> visualLatent, DenseTensor<float> faceIdentity, int steps)
    {
        if (faceIdentity.Length != _config.FaceLatentDim)
            throw new ArgumentException(
                $"faceIdentity has {faceIdentity.Length} elements; expected FaceLatentDim ({_config.FaceLatentDim}).");
        _net.Learn(visualLatent, faceIdentity, steps);
    }

    public void FlushAll() => _net.FlushStore();
    public void Dispose() => _net.Dispose();
}
