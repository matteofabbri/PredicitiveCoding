using PredictiveCoding.FaceRecognition.Descriptors;

namespace PredictiveCoding.FaceRecognition;

/// <summary>
/// Improved LBP embedder that adds two pre-processing stages before the descriptor:
/// <list type="bullet">
///   <item><b>Tan &amp; Triggs illumination normalization</b> — removes lighting gradients via DoG
///         and contrast equalization; the single biggest accuracy gain over raw LBP.</item>
///   <item><b>PCA whitening</b> (subspace power iteration) — decorrelates the 4704-D LBP
///         histogram and projects it down to a compact, spherically-distributed vector.</item>
/// </list>
/// Expected LFW accuracy: ~78–84%, vs. ~68–72% for raw LBP without preprocessing.
/// </summary>
public sealed class EnhancedLbpEmbedder : IFaceEmbedder
{
    private readonly int _targetDim;
    private readonly int _maxTrain;
    private readonly PcaProjector _pca = new();

    public EnhancedLbpEmbedder(int targetDim = 200, int maxTrain = 2000)
    {
        _targetDim = targetDim;
        _maxTrain  = maxTrain;
    }

    public int    Dimension => _pca.IsFit ? _pca.OutputDim : _targetDim;
    public string Name      => $"Enhanced LBP (DoG + PCA {_targetDim}-D)";

    public void Fit(IReadOnlyList<string> trainingImagePaths)
    {
        Console.WriteLine($"  [{Name}] Computing {Math.Min(trainingImagePaths.Count, _maxTrain)} descriptors for PCA …");
        var descriptors = new List<float[]>(Math.Min(trainingImagePaths.Count, _maxTrain));

        var rng  = new Random(7);
        var paths = trainingImagePaths.Count <= _maxTrain
            ? trainingImagePaths.ToList()
            : Subsample(trainingImagePaths, _maxTrain, rng);

        int done = 0;
        foreach (string path in paths)
        {
            try
            {
                byte[] raw = FaceDescriptor.LoadPixels(path);
                byte[] norm = FacePreprocessor.NormalizeTanTriggs(raw, 128);
                descriptors.Add(FaceDescriptor.ComputeFromPixels(norm));
            }
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
        byte[] raw  = FaceDescriptor.LoadPixels(imagePath);
        byte[] norm = FacePreprocessor.NormalizeTanTriggs(raw, 128);
        return _pca.Project(FaceDescriptor.ComputeFromPixels(norm));
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
