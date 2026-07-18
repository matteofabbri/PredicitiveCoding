using PredictiveCoding.FaceRecognition;
using PredictiveCoding.Imaging;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// ArcFace→PC distillation, offline: a fake teacher assigns each synthetic face
/// class a fixed target embedding; the PC student must learn the mapping, persist
/// it, and serve it back teacher-free. No ONNX models or datasets required.
/// </summary>
public sealed class PcDistilledFaceEmbedderTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"pcdistil_{Guid.NewGuid():N}");

    public PcDistilledFaceEmbedderTests() => Directory.CreateDirectory(_tmp);

    public void Dispose()
    {
        try { Directory.Delete(_tmp, true); } catch { /* FasterKV may hold logs briefly */ }
    }

    // ── Fixture: two visually distinct face classes + a deterministic teacher ──

    /// <summary>Class "a" = horizontal stripes, class "b" = checkerboard — very
    /// different LBP descriptors, so the mapping is learnable from few samples.</summary>
    private string WriteFace(string cls, int seed)
    {
        string path = Path.Combine(_tmp, $"{cls}_{seed}.png");
        var img = new RgbImage(64, 64);
        for (int y = 0; y < 64; y++)
            for (int x = 0; x < 64; x++)
            {
                bool on = cls == "a" ? (y / 4 + seed) % 2 == 0 : ((x / 4) + (y / 4) + seed) % 2 == 0;
                byte v = on ? (byte)230 : (byte)25;
                img[x, y] = new Rgb24(v, v, v);
            }
        img.SaveAsPng(path);
        return path;
    }

    private sealed class FakeTeacher : IFaceEmbedder
    {
        public static readonly float[] TargetA = Unit([1f, 0.3f, -0.5f, 0.2f]);
        public static readonly float[] TargetB = Unit([-0.6f, 0.8f, 0.4f, -0.3f]);

        public int Dimension => 4;
        public string Name => "fake-teacher";
        public void Fit(IReadOnlyList<string> trainingImagePaths) { }
        public float[] Embed(string imagePath)
            => (float[])(Path.GetFileName(imagePath).StartsWith('a') ? TargetA : TargetB).Clone();

        private static float[] Unit(float[] v)
        {
            float n = MathF.Sqrt(v.Sum(x => x * x));
            return [.. v.Select(x => x / n)];
        }
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return dot / Math.Max(Math.Sqrt(na) * Math.Sqrt(nb), 1e-12);
    }

    private List<string> TrainingSet()
    {
        var paths = new List<string>();
        for (int s = 0; s < 8; s++)
        {
            paths.Add(WriteFace("a", s));
            paths.Add(WriteFace("b", s));
        }
        return paths;
    }

    private PcDistilledFaceEmbedder NewStudent(string store, IFaceEmbedder? teacher) => new(
        store, teacher, hiddenDim: 32, inferenceSteps: 48, epochs: 30, maxTrain: 64);

    // ── Tests ───────────────────────────────────────────────────────────────────

    [Fact]
    public void Fit_DistilsTeacher_AndEmbeddingsFollowTheTargets()
    {
        string store = Path.Combine(_tmp, "store1");
        using var student = NewStudent(store, new FakeTeacher());
        student.Fit(TrainingSet());

        Assert.True(File.Exists(Path.Combine(store, "distilled.json")));
        Assert.Equal(4, student.Dimension);
        Assert.Contains("fake-teacher", student.Name);

        // Held-out images (seeds unseen in training).
        float[] embA = student.Embed(WriteFace("a", 99));
        float[] embB = student.Embed(WriteFace("b", 99));

        Assert.Equal(4, embA.Length);
        Assert.All(embA, v => Assert.True(float.IsFinite(v)));
        Assert.Equal(1.0, Math.Sqrt(embA.Sum(v => (double)v * v)), 3);   // L2-normalised

        // The student sends each class toward its teacher target.
        Assert.True(Cosine(embA, FakeTeacher.TargetA) > Cosine(embA, FakeTeacher.TargetB),
            $"class-a embedding closer to TargetB (a→A {Cosine(embA, FakeTeacher.TargetA):F3}, a→B {Cosine(embA, FakeTeacher.TargetB):F3})");
        Assert.True(Cosine(embB, FakeTeacher.TargetB) > Cosine(embB, FakeTeacher.TargetA),
            $"class-b embedding closer to TargetA (b→B {Cosine(embB, FakeTeacher.TargetB):F3}, b→A {Cosine(embB, FakeTeacher.TargetA):F3})");
    }

    [Fact]
    public void TrainedStore_ReloadsWithoutTeacher_AndEmbedsIdentically()
    {
        string store = Path.Combine(_tmp, "store2");
        string probe = WriteFace("a", 50);

        float[] first;
        using (var student = NewStudent(store, new FakeTeacher()))
        {
            student.Fit(TrainingSet());
            first = student.Embed(probe);
        }

        // Fresh instance, NO teacher: must load from disk and embed the same.
        using var reloaded = NewStudent(store, teacher: null);
        Assert.Equal(4, reloaded.Dimension);
        float[] again = reloaded.Embed(probe);   // no Fit needed either

        Assert.Equal(first.Length, again.Length);
        for (int i = 0; i < first.Length; i++)
            Assert.Equal(first[i], again[i], 4);
    }

    [Fact]
    public void Fit_OnTrainedStore_IsANoOp_ThatNeverTouchesTheTeacher()
    {
        string store = Path.Combine(_tmp, "store3");
        using (var student = NewStudent(store, new FakeTeacher()))
            student.Fit(TrainingSet());

        using var second = NewStudent(store, new ThrowingTeacher());
        second.Fit(TrainingSet());               // must load, not re-distil
        Assert.Equal(4, second.Embed(WriteFace("a", 77)).Length);
    }

    private sealed class ThrowingTeacher : IFaceEmbedder
    {
        public int Dimension => 4;
        public string Name => "must-not-be-called";
        public void Fit(IReadOnlyList<string> trainingImagePaths) => throw new InvalidOperationException();
        public float[] Embed(string imagePath) => throw new InvalidOperationException();
    }

    [Fact]
    public void ManifestWithoutCheckpoint_FailsLoudly_InsteadOfServingRandomWeights()
    {
        string store = Path.Combine(_tmp, "store5");
        using (var student = NewStudent(store, new FakeTeacher()))
            student.Fit(TrainingSet());

        Directory.Delete(Path.Combine(store, "checkpoints"), recursive: true);   // lost checkpoint

        using var broken = NewStudent(store, teacher: null);
        var ex = Assert.Throws<InvalidOperationException>(() => broken.Embed(WriteFace("a", 1)));
        Assert.Contains("does not match its manifest", ex.Message);
    }

    [Fact]
    public void LeftoverStoreWithoutManifest_IsPurged_AndRetrainedFresh()
    {
        string store = Path.Combine(_tmp, "store6");
        using (var student = NewStudent(store, new FakeTeacher()))
            student.Fit(TrainingSet());

        File.Delete(Path.Combine(store, "distilled.json"));   // e.g. crash window / manual reset

        using var retrained = NewStudent(store, new FakeTeacher());
        retrained.Fit(TrainingSet());                          // must retrain, not resume silently

        Assert.True(File.Exists(Path.Combine(store, "distilled.json")));
        float[] embA = retrained.Embed(WriteFace("a", 98));
        Assert.True(Cosine(embA, FakeTeacher.TargetA) > Cosine(embA, FakeTeacher.TargetB));
    }

    [Fact]
    public void FailedFit_LeavesInstanceRetryable()
    {
        string store = Path.Combine(_tmp, "store7");
        var flaky = new FlakyTeacher();
        using var student = NewStudent(store, flaky);

        Assert.Throws<InvalidOperationException>(() => student.Fit(TrainingSet()));   // pass 1: no pairs

        flaky.Failing = false;
        student.Fit(TrainingSet());                            // pass 2: teacher recovered
        Assert.Equal(4, student.Embed(WriteFace("a", 42)).Length);
    }

    private sealed class FlakyTeacher : IFaceEmbedder
    {
        public bool Failing { get; set; } = true;
        public int Dimension => 4;
        public string Name => "flaky-teacher";
        public void Fit(IReadOnlyList<string> trainingImagePaths) { }
        public float[] Embed(string imagePath)
        {
            if (Failing) throw new IOException("teacher offline");
            return (float[])(Path.GetFileName(imagePath).StartsWith('a')
                ? FakeTeacher.TargetA : FakeTeacher.TargetB).Clone();
        }
    }

    [Fact]
    public void Dispose_DisposesTheTeacher_OnlyWhenOwned()
    {
        var owned = new DisposableTeacher();
        var shared = new DisposableTeacher();
        new PcDistilledFaceEmbedder(Path.Combine(_tmp, "s8"), owned, disposeTeacher: true).Dispose();
        new PcDistilledFaceEmbedder(Path.Combine(_tmp, "s9"), shared).Dispose();

        Assert.True(owned.Disposed);
        Assert.False(shared.Disposed);
    }

    private sealed class DisposableTeacher : IFaceEmbedder, IDisposable
    {
        public bool Disposed { get; private set; }
        public int Dimension => 4;
        public string Name => "disposable-teacher";
        public void Fit(IReadOnlyList<string> trainingImagePaths) { }
        public float[] Embed(string imagePath) => new float[4];
        public void Dispose() => Disposed = true;
    }

    [Fact]
    public void MissingStoreAndTeacher_FailLoudly()
    {
        string store = Path.Combine(_tmp, "store4");

        using (var noTeacher = NewStudent(store, teacher: null))
        {
            Assert.Throws<InvalidOperationException>(() => noTeacher.Fit(TrainingSet()));
            Assert.Throws<InvalidOperationException>(() => noTeacher.Embed(WriteFace("a", 1)));
            Assert.Throws<InvalidOperationException>(() => _ = noTeacher.Dimension);
        }

        using var unusable = NewStudent(store, new FakeTeacher());
        Assert.Throws<InvalidOperationException>(() => unusable.Fit([]));   // no usable pairs
    }
}
