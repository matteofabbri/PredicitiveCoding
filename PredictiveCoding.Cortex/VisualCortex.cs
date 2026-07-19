using PredictiveCoding.Core;
using PredictiveCoding.Networks.Hierarchical;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// VisualCortex  —  hierarchical predictive-coding model of the ventral stream
//
//   image [C,H,W]
//     │  RetinotopicMap : tile into P×P receptive fields
//     ▼
//   V1  : one weight-shared cortical column (PCN) encodes every patch → K features
//     │  concatenate over the GridH×GridW grid → V1 feature map
//     ▼
//   V2…IT : HierarchicalNetwork (deep modular hierarchy) → abstract object latent
//     │
//     ▼
//   Readout (optional) : PCN classifier on the IT latent
//
// All learning is local predictive coding (no backprop):
//   • the shared V1 column learns Gabor-like receptive fields by autoencoding
//     every patch it sees (Rao & Ballard 1999), with weights shared across all
//     retinotopic positions (= convolution);
//   • each higher area greedily learns to encode the area below it;
//   • the optional readout learns IT → class by min-energy classification.
//
// References:
//   Rao & Ballard (1999); Felleman & Van Essen (1991); Friston (2005).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class VisualCortex : IDisposable
{
    private readonly VisualCortexConfig  _config;
    private readonly RetinotopicMap      _map;
    private readonly Network             _v1Column;   // weight-shared across positions
    private readonly HierarchicalNetwork _higher;     // V2 → V4 → IT
    private readonly Network?            _classifier; // optional readout

    public VisualCortexConfig Configuration => _config;
    public RetinotopicMap     Retinotopy    => _map;

    /// <summary>Number of V1 feature maps K (cortical-column output size).</summary>
    public int V1FeatureDim => _config.V1FeatureDim;

    /// <summary>Dimensionality of the IT (top-area) object latent.</summary>
    public int ItLatentDim => _config.ItLatentDim;

    /// <summary>True when a readout classifier is configured.</summary>
    public bool HasClassifier => _classifier is not null;

    public VisualCortex(VisualCortexConfig config, Random rng)
    {
        config.Validate();
        _config = config;

        _map = new RetinotopicMap(
            config.ImageChannels, config.ImageHeight, config.ImageWidth,
            config.PatchSize, config.PatchStride);

        var v1Cfg = config.V1Column with { Path = Path.Combine(config.BasePath, "v1") };
        _v1Column = new Network(v1Cfg, new Random(rng.Next()));

        var higherCfg = config.HigherAreas with { BasePath = Path.Combine(config.BasePath, "higher") };
        _higher = new HierarchicalNetwork(higherCfg, new Random(rng.Next()));

        if (config.Classifier is not null)
        {
            var clsCfg = config.Classifier with { Path = Path.Combine(config.BasePath, "readout") };
            _classifier = new Network(clsCfg, new Random(rng.Next()));
        }
    }

    // ── Encoding ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Builds the V1 feature map: encodes every receptive-field patch with the
    /// shared cortical column and concatenates the K-dim responses over the grid.
    /// Returns a vector of length GridH·GridW·K.
    /// </summary>
    public DenseTensor<float> EncodeV1(DenseTensor<float> image, int inferenceSteps)
    {
        var patches = _map.Extract(image);
        int k       = V1FeatureDim;
        var fm      = new DenseTensor<float>(_config.FeatureMapDim);

        for (int p = 0; p < patches.Length; p++)
        {
            var resp = _v1Column.InferGetTop(patches[p], inferenceSteps);
            Array.Copy(resp.Buffer, 0, fm.Buffer, p * k, k);
        }
        return fm;
    }

    /// <summary>Encodes an image all the way to the IT object latent.</summary>
    public DenseTensor<float> Encode(DenseTensor<float> image, int inferenceSteps)
        => _higher.Encode(EncodeV1(image, inferenceSteps), inferenceSteps);

    /// <summary>
    /// Returns the latent of every cortical area: index 0 = V1 feature map,
    /// followed by each higher area (V2, V4, …, IT) in ascending abstraction.
    /// </summary>
    public DenseTensor<float>[] EncodeAreas(DenseTensor<float> image, int inferenceSteps)
    {
        var v1     = EncodeV1(image, inferenceSteps);
        var higher = _higher.EncodeAll(v1, inferenceSteps);
        var areas  = new DenseTensor<float>[higher.Length + 1];
        areas[0]   = v1;
        Array.Copy(higher, 0, areas, 1, higher.Length);
        return areas;
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>
    /// Unsupervised learning on one image:
    ///   1. The shared V1 column autoencodes every receptive-field patch
    ///      (weights shared across positions ⇒ it learns position-invariant
    ///      local features).
    ///   2. The V1 feature map is built and the higher-area hierarchy greedily
    ///      learns to encode it.
    /// </summary>
    public void Learn(DenseTensor<float> image, int inferenceSteps)
    {
        var patches = _map.Extract(image);
        foreach (var patch in patches)
            _v1Column.LearnUnsupervised(patch, inferenceSteps);

        var fm = EncodeV1(image, inferenceSteps);
        _higher.Learn(fm, inferenceSteps);
    }

    /// <summary>
    /// Supervised learning: runs the unsupervised cortex update, then trains the
    /// readout classifier on the IT latent.  Requires a configured classifier.
    /// </summary>
    public void LearnSupervised(DenseTensor<float> image, DenseTensor<float> label, int inferenceSteps)
    {
        if (_classifier is null)
            throw new InvalidOperationException(
                "LearnSupervised requires a Classifier in the VisualCortexConfig.");

        Learn(image, inferenceSteps);
        var it = Encode(image, inferenceSteps);
        _classifier.Learn(it, label, inferenceSteps);
    }

    // ── Recognition ────────────────────────────────────────────────────────────

    /// <summary>Classifies an image via the IT latent and the readout classifier.</summary>
    public int Predict(DenseTensor<float> image, int inferenceSteps)
    {
        if (_classifier is null)
            throw new InvalidOperationException(
                "Predict requires a Classifier in the VisualCortexConfig.");

        var it = Encode(image, inferenceSteps);
        return _classifier.Predict(it, inferenceSteps);
    }

    /// <summary>Classification with the winning free energy for diagnostics.</summary>
    public (int cls, float energy) PredictWithEnergy(DenseTensor<float> image, int inferenceSteps)
    {
        if (_classifier is null)
            throw new InvalidOperationException(
                "PredictWithEnergy requires a Classifier in the VisualCortexConfig.");

        var it = Encode(image, inferenceSteps);
        return _classifier.PredictWithEnergy(it, inferenceSteps);
    }

    // ── Generation ───────────────────────────────────────────────────────────

    /// <summary>
    /// Predictive reconstruction of the image: encode to IT, regenerate the V1
    /// feature map top-down through the hierarchy, then have the shared column
    /// generate each receptive-field patch and reassemble them (overlap averaged).
    /// </summary>
    public DenseTensor<float> Reconstruct(DenseTensor<float> image, int inferenceSteps)
    {
        var fm    = EncodeV1(image, inferenceSteps);
        var fmHat = _higher.Reconstruct(fm, inferenceSteps);

        int k          = V1FeatureDim;
        var patchRecon = new DenseTensor<float>[_map.PatchCount];
        for (int p = 0; p < _map.PatchCount; p++)
        {
            var feat = new DenseTensor<float>(k);
            Array.Copy(fmHat.Buffer, p * k, feat.Buffer, 0, k);
            patchRecon[p] = _v1Column.Generate(feat, inferenceSteps);
        }
        return _map.Assemble(patchRecon);
    }

    /// <summary>Half mean-squared reconstruction error ½‖x − Reconstruct(x)‖².</summary>
    public float ReconstructionError(DenseTensor<float> image, int inferenceSteps)
    {
        var x_hat = Reconstruct(image, inferenceSteps);
        return 0.5f * TensorUtils.SumSquares(TensorUtils.ElementwiseSub(image, x_hat));
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    /// <summary>PC free energy of the V1 column after its last inference.</summary>
    public float V1Energy() => _v1Column.ComputeEnergy();

    /// <summary>PC free energy of higher area <paramref name="areaIndex"/> (0 = V2).</summary>
    public float HigherAreaEnergy(int areaIndex) => _higher.ModuleEnergy(areaIndex);

    public void FlushAll()
    {
        _v1Column.FlushStore();
        _higher.FlushAll();
        _classifier?.FlushStore();
    }

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose()
    {
        _v1Column.Dispose();
        _higher.Dispose();
        _classifier?.Dispose();
    }
}
