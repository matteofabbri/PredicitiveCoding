using PredictiveCoding.Imaging;
using PredictiveCoding.Vision.Detectors;

namespace PredictiveCoding.Vision.ReId;

/// <summary>
/// Turns a person's image region into a fixed-length appearance vector used to tell
/// people apart for re-identification. Implementations crop the detection box from
/// the frame and return an L2-normalised embedding, so two crops of the same person
/// have a high cosine similarity and two different people a low one. The bundled
/// <see cref="ColorHistogramEmbedder"/> is a dependency-free baseline; a face
/// (ArcFace) or body Re-ID (OSNet ONNX) model can implement the same interface.
/// </summary>
public interface IAppearanceEmbedder
{
    /// <summary>Human-readable embedder name (for logging / gallery compatibility).</summary>
    string Name { get; }

    /// <summary>Length of the embedding vectors produced.</summary>
    int Dimension { get; }

    /// <summary>Embeds the image region described by <paramref name="box"/> taken from
    /// <paramref name="frame"/>. The result is L2-normalised (or all-zero for an empty
    /// region).</summary>
    float[] Embed(RgbImage frame, Detection box);
}
