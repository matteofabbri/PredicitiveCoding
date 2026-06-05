namespace PredictiveCoding.FaceRecognition;

/// <summary>
/// Turns a face image into a vector. <see cref="Fit"/> lets learnable embedders (e.g. the Predictive Coding
/// one) train on a sample of images first; for fixed descriptors it is a no-op.
/// </summary>
public interface IFaceEmbedder
{
    /// <summary>Dimensionality of the produced vectors.</summary>
    int Dimension { get; }

    /// <summary>Short human-readable name (shown in benchmark output).</summary>
    string Name { get; }

    /// <summary>Optionally train the embedder on a representative sample of image paths.</summary>
    void Fit(IReadOnlyList<string> trainingImagePaths);

    /// <summary>Embeds a single image.</summary>
    float[] Embed(string imagePath);

    /// <summary>
    /// Embeds many images at once. Deep embedders override this to run a real GPU batch; the default just loops
    /// <see cref="Embed"/>. Entries are <see langword="null"/> for images that could not be read.
    /// </summary>
    IReadOnlyList<float[]?> EmbedBatch(IReadOnlyList<string> imagePaths)
    {
        var result = new float[]?[imagePaths.Count];
        for (int i = 0; i < imagePaths.Count; i++)
        {
            try { result[i] = Embed(imagePaths[i]); }
            catch { result[i] = null; }
        }
        return result;
    }
}

/// <summary>The raw <see cref="FaceDescriptor"/> used directly as the embedding (a fixed baseline).</summary>
public sealed class DescriptorEmbedder : IFaceEmbedder
{
    public int Dimension => FaceDescriptor.Dimension;
    public string Name => "LBP descriptor";
    public void Fit(IReadOnlyList<string> trainingImagePaths) { }
    public float[] Embed(string imagePath) => FaceDescriptor.Compute(imagePath);
}
