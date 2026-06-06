using PredictiveCoding.FaceRecognition.Descriptors;

namespace PredictiveCoding.FaceRecognition;

/// <summary>
/// Fused LBP + HOG embedder with PCA whitening.
/// <para>
/// Each face image is described by the <b>concatenation</b> of two complementary descriptors:
/// <list type="bullet">
///   <item><b>LBP</b> (4704-D) — captures local binary texture (illumination-tolerant).</item>
///   <item><b>HOG</b> (1568-D) — captures oriented gradient structure (shape/edge).</item>
/// </list>
/// The 6272-D fused descriptor is projected to <c>targetDim</c> via whitened PCA.
/// Both descriptors are computed on a gamma-corrected image so that HOG gradients
/// reflect meaningful intensity changes.
/// </para>
/// Expected LFW accuracy: ~82–87%, notably above the enhanced-LBP baseline (~78–84%).
/// </summary>
public sealed class FusedEmbedder : IFaceEmbedder
{
    private readonly int _targetDim;
    private readonly int _maxTrain;
    private readonly PcaProjector _pca = new();

    public FusedEmbedder(int targetDim = 256, int maxTrain = 2000)
    {
        _targetDim = targetDim;
        _maxTrain  = maxTrain;
    }

    public int    Dimension => _pca.IsFit ? _pca.OutputDim : _targetDim;
    public string Name      => $"Fused LBP+HOG (PCA {_targetDim}-D)";

    public void Fit(IReadOnlyList<string> trainingImagePaths)
    {
        Console.WriteLine($"  [{Name}] Computing {Math.Min(trainingImagePaths.Count, _maxTrain)} fused descriptors …");
        var descriptors = new List<float[]>(Math.Min(trainingImagePaths.Count, _maxTrain));

        var rng   = new Random(7);
        var paths = trainingImagePaths.Count <= _maxTrain
            ? trainingImagePaths.ToList()
            : Subsample(trainingImagePaths, _maxTrain, rng);

        int done = 0;
        foreach (string path in paths)
        {
            try   { descriptors.Add(ComputeFused(path)); }
            catch { /* skip unreadable images */ }

            if (++done % 500 == 0)
                Console.WriteLine($"    {done}/{paths.Count} …");
        }

        _pca.Fit(descriptors, _targetDim, maxSamples: _maxTrain,
            onProgress: s => Console.WriteLine("  " + s));
    }

    public float[] Embed(string imagePath)
    {
        if (!_pca.IsFit)
            throw new InvalidOperationException("Call Fit before Embed.");
        return _pca.Project(ComputeFused(imagePath));
    }

    public IReadOnlyList<float[]?> EmbedBatch(IReadOnlyList<string> paths)
    {
        var result = new float[]?[paths.Count];
        Parallel.For(0, paths.Count, i =>
        {
            try   { result[i] = Embed(paths[i]); }
            catch { result[i] = null; }
        });
        return result;
    }

    // ── Descriptor ────────────────────────────────────────────────────────────────

    private static float[] ComputeFused(string imagePath)
    {
        // Share the image load between LBP and HOG; gamma-correct once.
        byte[] raw   = FaceDescriptor.LoadPixels(imagePath);
        byte[] gamma = FacePreprocessor.GammaCorrect(raw, 128);

        float[] lbp = FaceDescriptor.ComputeFromPixels(gamma);  // 4704-D
        float[] hog = HogDescriptor.Compute(gamma, 128);        // 1568-D

        // Concatenate: 4704 + 1568 = 6272-D
        var fused = new float[lbp.Length + hog.Length];
        lbp.CopyTo(fused.AsSpan());
        hog.CopyTo(fused.AsSpan(lbp.Length));
        return fused;
    }

    private static List<string> Subsample(IReadOnlyList<string> items, int n, Random rng)
    {
        var idx = Enumerable.Range(0, items.Count).ToList();
        for (int i = idx.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (idx[i], idx[j]) = (idx[j], idx[i]);
        }
        return idx.Take(n).Select(i => items[i]).ToList();
    }
}
