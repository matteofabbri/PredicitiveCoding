using PredictiveCoding.Core;
using PredictiveCoding.Expanded.GenerativeNetwork;
using PredictiveCoding.Networks;

namespace PredictiveCoding.FaceRecognition;

/// <summary>
/// A face embedder learned with <b>Predictive Coding</b> (no backprop). A <see cref="GenerativeNetwork"/> — a
/// PC VAE-like autoencoder — is trained, unsupervised, to reconstruct the <see cref="FaceDescriptor"/>s; the
/// latent code <c>z = Encode(x)</c> is the embedding. Unsupervised, so it works for open-set verification
/// (no identity labels) and identification alike.
/// </summary>
public sealed class PcFaceEmbedder : IFaceEmbedder, IDisposable
{
    private readonly int _latentDim;
    private readonly int _inferenceSteps;
    private readonly int _epochs;
    private readonly int _maxTrain;
    private readonly string _storeDir;
    private GenerativeNetwork? _network;

    public PcFaceEmbedder(string storeDir, int latentDim = 128, int inferenceSteps = 8, int epochs = 8, int maxTrain = 2000)
    {
        _storeDir = storeDir;
        _latentDim = latentDim;
        _inferenceSteps = inferenceSteps;
        _epochs = epochs;
        _maxTrain = maxTrain;
    }

    public int Dimension => _latentDim;
    public string Name => $"Predictive Coding latent ({_latentDim}-D)";

    public void Fit(IReadOnlyList<string> trainingImagePaths)
    {
        Directory.CreateDirectory(_storeDir);
        int inputDim = FaceDescriptor.Dimension;

        var config = new GenerativeNetworkConfig
        {
            BasePath = _storeDir,
            PriorStrength = 0.1f,
            NetworkConfig = new NetworkConfig
            {
                LayerSizes = [inputDim, 128, _latentDim],
                Activation = ActivationType.Tanh,
                StateLearningRate = 0.1f,
                WeightLearningRate = 0.01f,
                LazyPersistence = true,
            },
        };

        _network?.Dispose();
        _network = new GenerativeNetwork(config, new Random(1));

        // Train on a capped, descriptor-encoded sample (unsupervised reconstruction).
        var sample = Sample(trainingImagePaths, _maxTrain, new Random(7));
        var descriptors = new List<float[]>(sample.Count);
        foreach (string path in sample)
        {
            try { descriptors.Add(FaceDescriptor.Compute(path)); }
            catch { /* skip unreadable images */ }
        }

        var order = Enumerable.Range(0, descriptors.Count).ToArray();
        var rng = new Random(11);
        for (int epoch = 0; epoch < _epochs; epoch++)
        {
            for (int i = order.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (order[i], order[j]) = (order[j], order[i]);
            }
            foreach (int i in order)
                _network.Learn(new DenseTensor<float>((float[])descriptors[i].Clone(), inputDim), _inferenceSteps);
            _network.FlushStore();
        }
    }

    public float[] Embed(string imagePath)
    {
        if (_network is null)
            throw new InvalidOperationException("Call Fit before Embed.");
        float[] descriptor = FaceDescriptor.Compute(imagePath);
        DenseTensor<float> latent = _network.Encode(new DenseTensor<float>(descriptor, FaceDescriptor.Dimension), _inferenceSteps);
        return (float[])latent.Buffer.Clone();
    }

    private static List<string> Sample(IReadOnlyList<string> items, int max, Random rng)
    {
        if (items.Count <= max)
            return [.. items];
        var copy = items.ToArray();
        for (int i = copy.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (copy[i], copy[j]) = (copy[j], copy[i]);
        }
        return [.. copy.Take(max)];
    }

    public void Dispose() => _network?.Dispose();
}
