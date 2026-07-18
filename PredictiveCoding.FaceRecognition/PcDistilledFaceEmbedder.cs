using System.Text.Json;
using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.FaceRecognition;

/// <summary>
/// A face embedder <b>distilled</b> from a teacher (typically <see cref="ArcFaceOnnxEmbedder"/>)
/// into a pure <b>Predictive Coding</b> network — no backprop, no ONNX at inference.
/// <para>
/// Training is supervised PC regression: for each face, layer 0 is clamped to the
/// <see cref="FaceDescriptor"/> (LBP) and the top layer is clamped to the teacher's
/// embedding; iterative inference settles the hidden states and Hebbian updates move
/// the generative weights (<see cref="Network.Learn"/>). At inference the embedding is
/// the free-settled top-layer state (<see cref="Network.InferGetTop"/>), L2-normalised
/// like the teacher's.
/// </para>
/// The teacher is only needed the first time: after <see cref="Fit"/> the trained
/// network persists in <c>storeDir</c> (LayerStore checkpoint + <c>distilled.json</c>
/// manifest) and later instances embed teacher-free — the distilled model is 100% in-house.
/// </summary>
public sealed class PcDistilledFaceEmbedder : IFaceEmbedder, IDisposable
{
    private const int ManifestVersion = 1;
    private const string ManifestFile = "distilled.json";
    private const int ProbeLength = 8;          // embedding prefix of a fixed reference input
    private const float ProbeTolerance = 1e-3f;

    /// <summary>Probe = the first values of the trained network's embedding of a fixed
    /// reference descriptor. On load it is recomputed and compared: a missing or
    /// corrupt LayerStore checkpoint silently reseeds RANDOM weights, and without
    /// this check such a store would serve garbage embeddings without any error.</summary>
    private sealed record Manifest(
        int Version, int InputDim, int HiddenDim, int OutputDim,
        int InferenceSteps, string TeacherName, int SampleCount, int Epochs, float[] Probe);

    private readonly string _storeDir;
    private readonly IFaceEmbedder? _teacher;
    private readonly bool _disposeTeacher;
    private readonly int _hiddenDim;
    private readonly int _inferenceSteps;
    private readonly int _epochs;
    private readonly int _maxTrain;

    private Network? _network;
    private Manifest? _manifest;

    /// <param name="storeDir">Directory holding the trained network (LayerStore + manifest).
    /// If it already contains a distilled model, it is reused and the teacher is not needed.</param>
    /// <param name="teacher">Teacher embedder to distil from on the first <see cref="Fit"/>
    /// (e.g. <see cref="ArcFaceOnnxEmbedder"/>). May be null when the store is already trained.</param>
    /// <param name="disposeTeacher">When true, <see cref="Dispose"/> also disposes the
    /// teacher — for callers that hand ownership over (e.g. a teacher created inline).</param>
    public PcDistilledFaceEmbedder(
        string storeDir,
        IFaceEmbedder? teacher = null,
        int hiddenDim = 1024,
        int inferenceSteps = 8,
        int epochs = 10,
        int maxTrain = 2000,
        bool disposeTeacher = false)
    {
        ArgumentException.ThrowIfNullOrEmpty(storeDir);
        ArgumentOutOfRangeException.ThrowIfLessThan(hiddenDim, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(inferenceSteps, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(epochs, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(maxTrain, 1);
        _storeDir = storeDir;
        _teacher = teacher;
        _disposeTeacher = disposeTeacher;
        _hiddenDim = hiddenDim;
        _inferenceSteps = inferenceSteps;
        _epochs = epochs;
        _maxTrain = maxTrain;
    }

    public int Dimension => (_manifest ?? PeekManifest())?.OutputDim ?? _teacher?.Dimension
        ?? throw new InvalidOperationException(
            $"Embedding dimension unknown: no trained store in '{_storeDir}' and no teacher supplied.");

    public string Name =>
        $"PC distilled ← {(_manifest ?? PeekManifest())?.TeacherName ?? _teacher?.Name ?? "?"}";

    /// <summary>
    /// Distils the teacher on the first call: teacher embeddings become regression
    /// targets for the PC network. When <c>storeDir</c> already holds a trained model
    /// this is a cheap no-op (the model is loaded, the teacher never touched).
    /// </summary>
    public void Fit(IReadOnlyList<string> trainingImagePaths)
    {
        if (_network is not null || TryLoadExisting())
            return;
        if (_teacher is null)
            throw new InvalidOperationException(
                $"No trained model in '{_storeDir}' and no teacher supplied — pass a teacher " +
                "embedder (e.g. ArcFaceOnnxEmbedder) to distil first.");
        ArgumentNullException.ThrowIfNull(trainingImagePaths);

        Directory.CreateDirectory(_storeDir);
        int inputDim = FaceDescriptor.Dimension;

        // 1. The teacher labels a capped sample: embedding = regression target.
        List<string> sample = SampleShuffled(trainingImagePaths, _maxTrain, new Random(7));
        _teacher.Fit(sample);
        IReadOnlyList<float[]?> targets = _teacher.EmbedBatch(sample);

        var pairs = new List<(float[] Descriptor, float[] Target)>(sample.Count);
        for (int i = 0; i < sample.Count; i++)
        {
            if (targets[i] is not float[] target || target.Length != _teacher.Dimension)
                continue;
            try
            {
                pairs.Add((FaceDescriptor.Compute(sample[i]), target));
            }
            catch
            {
                // unreadable image: the teacher may still have embedded a cached copy — skip
            }
        }
        if (pairs.Count == 0)
            throw new InvalidOperationException(
                "Distillation produced no usable (descriptor, teacher-embedding) pairs.");

        // 2. Supervised PC training: clamp input to the descriptor, top to the target.
        //    No manifest ⇒ untrained: purge any leftover LayerStore files first, or a
        //    stale checkpoint (crash window, hand-deleted manifest) would be silently
        //    recovered and training would resume on old — possibly mis-sized — weights.
        PurgeStoreFiles(_storeDir);
        var manifest = new Manifest(ManifestVersion, inputDim, _hiddenDim, _teacher.Dimension,
            _inferenceSteps, _teacher.Name, pairs.Count, _epochs, Probe: []);
        try
        {
            _network?.Dispose();
            _network = OpenNetwork(manifest);

            DenseTensor<float>[] inputs = [.. pairs.Select(p => new DenseTensor<float>(p.Descriptor, inputDim))];
            DenseTensor<float>[] labels = [.. pairs.Select(p => new DenseTensor<float>(p.Target, manifest.OutputDim))];
            int[] order = Enumerable.Range(0, pairs.Count).ToArray();
            var rng = new Random(11);
            for (int epoch = 0; epoch < _epochs; epoch++)
            {
                for (int i = order.Length - 1; i > 0; i--)
                {
                    int j = rng.Next(i + 1);
                    (order[i], order[j]) = (order[j], order[i]);
                }
                foreach (int i in order)
                    _network.Learn(inputs[i], labels[i], _inferenceSteps);
                _network.FlushStore();
            }

            // 3. Make it durable NOW: the LayerStore checkpoint is only taken on
            //    Dispose, so cycle the network — a crash can no longer lose training.
            _network.Dispose();
            _network = OpenNetwork(manifest);

            manifest = manifest with { Probe = ComputeProbe(manifest) };
            File.WriteAllText(Path.Combine(_storeDir, ManifestFile), JsonSerializer.Serialize(manifest));
            _manifest = manifest;
        }
        catch
        {
            // A half-trained store must not look trained: drop it entirely so a
            // retry (or a later instance) starts clean instead of silently serving
            // or resuming a partially trained network.
            _network?.Dispose();
            _network = null;
            PurgeStoreFiles(_storeDir);
            throw;
        }
    }

    /// <summary>Embeds a face: LBP descriptor → free PC inference → L2-normalised top state.
    /// Works without the teacher whenever <c>storeDir</c> holds a trained model.</summary>
    public float[] Embed(string imagePath)
    {
        if (_network is null && !TryLoadExisting())
            throw new InvalidOperationException("Call Fit before Embed (or point storeDir at a trained store).");

        return EmbedDescriptor(FaceDescriptor.Compute(imagePath), _manifest?.InferenceSteps ?? _inferenceSteps);
    }

    private float[] EmbedDescriptor(float[] descriptor, int steps)
    {
        // resetState: the embedding must be a function of THIS input only, not of
        // whatever top-layer state the previous sample (or training) left behind.
        DenseTensor<float> top = _network!.InferGetTop(
            new DenseTensor<float>(descriptor, descriptor.Length), steps, resetState: true);

        var embedding = (float[])top.Buffer.Clone();
        Normalize(embedding);
        return embedding;
    }

    // ── Persistence ─────────────────────────────────────────────────────────────

    private Manifest? PeekManifest()
    {
        string path = Path.Combine(_storeDir, ManifestFile);
        if (!File.Exists(path)) return null;
        try { return JsonSerializer.Deserialize<Manifest>(File.ReadAllText(path)); }
        catch { return null; }
    }

    private bool TryLoadExisting()
    {
        Manifest? m = PeekManifest();
        if (m is null || m.Version != ManifestVersion ||
            m.InputDim < 1 || m.HiddenDim < 1 || m.OutputDim < 1 || m.InferenceSteps < 1 ||
            m.Probe is not { Length: > 0 })
            return false;
        if (m.InputDim != FaceDescriptor.Dimension)
            throw new InvalidOperationException(
                $"Store '{_storeDir}' was distilled with a {m.InputDim}-D descriptor but " +
                $"FaceDescriptor now produces {FaceDescriptor.Dimension}-D — retrain into a fresh storeDir.");

        _network = OpenNetwork(m);   // recovers the LayerStore checkpoint automatically

        // A lost/corrupt checkpoint makes LayerStore silently seed RANDOM weights —
        // verify the recovered network still reproduces the manifest's probe.
        float[] probe = ComputeProbe(m);
        for (int i = 0; i < m.Probe.Length; i++)
        {
            if (Math.Abs(probe[i] - m.Probe[i]) > ProbeTolerance)
            {
                _network.Dispose();
                _network = null;
                throw new InvalidOperationException(
                    $"Store '{_storeDir}' does not match its manifest (LayerStore checkpoint " +
                    "missing or corrupt) — delete the directory and retrain.");
            }
        }

        _manifest = m;
        return true;
    }

    /// <summary>Embedding prefix of a fixed, deterministic reference descriptor —
    /// a cheap fingerprint of the trained weights.</summary>
    private float[] ComputeProbe(Manifest m)
    {
        var reference = new float[m.InputDim];
        var rng = new Random(42);
        for (int i = 0; i < reference.Length; i++) reference[i] = (float)rng.NextDouble();
        Normalize(reference);
        return [.. EmbedDescriptor(reference, m.InferenceSteps).Take(ProbeLength)];
    }

    private static void PurgeStoreFiles(string dir)
    {
        if (!Directory.Exists(dir)) return;
        string checkpoints = Path.Combine(dir, "checkpoints");
        if (Directory.Exists(checkpoints)) Directory.Delete(checkpoints, recursive: true);
        foreach (string file in Directory.EnumerateFiles(dir, "layers.log*"))
            File.Delete(file);
        string manifest = Path.Combine(dir, ManifestFile);
        if (File.Exists(manifest)) File.Delete(manifest);
    }

    private Network OpenNetwork(Manifest m) => new(
        new NetworkConfig
        {
            Path = _storeDir,
            LayerSizes = [m.InputDim, m.HiddenDim, m.OutputDim],
            Activation = ActivationType.Tanh,
            StateLearningRate = 0.1f,
            WeightLearningRate = 0.01f,
            LazyPersistence = true,
        },
        new Random(1));

    // ── Helpers ─────────────────────────────────────────────────────────────────

    private static List<string> SampleShuffled(IReadOnlyList<string> items, int max, Random rng)
    {
        var copy = items.ToArray();
        for (int i = copy.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (copy[i], copy[j]) = (copy[j], copy[i]);
        }
        return [.. copy.Take(max)];
    }

    private static void Normalize(float[] v)
    {
        double norm = 0;
        foreach (float x in v) norm += (double)x * x;
        norm = Math.Sqrt(norm);
        if (norm < 1e-10) return;   // degenerate all-zero state: leave as-is
        for (int i = 0; i < v.Length; i++) v[i] = (float)(v[i] / norm);
    }

    public void Dispose()
    {
        _network?.Dispose();
        if (_disposeTeacher && _teacher is IDisposable owned)
            owned.Dispose();
    }
}
