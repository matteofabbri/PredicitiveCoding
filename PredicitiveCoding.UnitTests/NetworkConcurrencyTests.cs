using PredictiveCoding.Core;
using PredictiveCoding.Model;

namespace PredicitiveCoding.UnitTests;

// ─────────────────────────────────────────────────────────────────────────────
// NetworkConcurrencyTests
//
// Verifies that a single Network instance is thread-safe: Learn and Predict
// can be called concurrently from multiple threads without data corruption,
// exceptions, or NaN/Inf values.
//
// Network is thread-safe by design (ReaderWriterLockSlim at the Network level
// + internal Lock in LayerStore).  These tests stress that guarantee.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class NetworkConcurrencyTests : IDisposable
{
    private readonly string _tempDir;

    public NetworkConcurrencyTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"pcnet_conc_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private Network BuildNetwork(string sub, int seed = 0)
    {
        var cfg = new NetworkConfig
        {
            Path               = Path.Combine(_tempDir, sub),
            LayerSizes         = [4, 8, 2],
            Activation         = ActivationType.Tanh,
            StateLearningRate  = 0.1f,
            WeightLearningRate = 0.01f,
        };
        return new Network(cfg, new Random(seed));
    }

    private static DenseTensor<float> Input(float a, float b, float c, float d) =>
        new(new[] { a, b, c, d }, 4);

    private static DenseTensor<float> OneHot(int index, int size = 2)
    {
        var t = DenseTensor<float>.Zeros(size);
        t.Buffer[index] = 1f;
        return t;
    }

    private static void RunThreads(int count, Action<int> body)
    {
        var errors  = new List<Exception>();
        var barrier = new Barrier(count);

        var threads = Enumerable.Range(0, count).Select(i => new Thread(() =>
        {
            barrier.SignalAndWait();
            try   { body(i); }
            catch (Exception ex) { lock (errors) errors.Add(ex); }
        })).ToList();

        threads.ForEach(t => t.Start());
        threads.ForEach(t => t.Join(TimeSpan.FromSeconds(30)));

        Assert.Empty(errors);
    }

    // ── Concurrent Learn ──────────────────────────────────────────────────────

    [Fact]
    public void Learn_ConcurrentCalls_NoExceptions()
    {
        using var net = BuildNetwork("learn_concurrent");
        var x0 = Input(1f, 0f, 0f, 0f);
        var x1 = Input(0f, 1f, 0f, 0f);
        var y0  = OneHot(0);
        var y1  = OneHot(1);

        RunThreads(6, i =>
        {
            var (x, y) = i % 2 == 0 ? (x0, y0) : (x1, y1);
            for (int step = 0; step < 30; step++)
                net.Learn(x, y, inferenceSteps: 3);
        });
    }

    [Fact]
    public void Learn_ConcurrentCalls_WeightNormRemainsFinite()
    {
        using var net = BuildNetwork("learn_norm");
        var x = Input(1f, 0f, 0f, 0f);
        var y = OneHot(0);

        RunThreads(4, _ =>
        {
            for (int step = 0; step < 20; step++)
                net.Learn(x, y, inferenceSteps: 5);
        });

        float norm = net.AverageWeightNorm();
        Assert.True(float.IsFinite(norm) && norm > 0f, $"WeightNorm: {norm}");
    }

    // ── Concurrent Predict ────────────────────────────────────────────────────

    [Fact]
    public void Predict_ConcurrentCalls_AllReturnValidClass()
    {
        using var net = BuildNetwork("predict_concurrent");
        var x = Input(0.5f, 0.1f, 0.9f, 0.3f);

        for (int i = 0; i < 20; i++)
            net.Learn(x, OneHot(0), inferenceSteps: 5);

        var results = new int[6];

        RunThreads(6, i =>
        {
            results[i] = net.Predict(x, inferenceSteps: 5);
        });

        Assert.All(results, cls => Assert.InRange(cls, 0, 1));
    }

    // ── Concurrent Learn + Predict ────────────────────────────────────────────

    [Fact]
    public void LearnAndPredict_Concurrent_NoExceptionsAndEnergyIsFinite()
    {
        using var net = BuildNetwork("learn_predict");
        var x0 = Input(0f, 0f, 0f, 0f);
        var x1 = Input(1f, 1f, 1f, 1f);
        var y0  = OneHot(0);
        var y1  = OneHot(1);

        // 3 learner threads + 3 predictor threads, all on the same network
        RunThreads(6, i =>
        {
            bool isLearner = i < 3;
            var (x, y) = i % 2 == 0 ? (x0, y0) : (x1, y1);

            for (int iter = 0; iter < 40; iter++)
            {
                if (isLearner)
                    net.Learn(x, y, inferenceSteps: 3);
                else
                    Assert.InRange(net.Predict(x, inferenceSteps: 3), 0, 1);
            }
        });

        float energy = net.ComputeEnergy();
        Assert.True(float.IsFinite(energy) && energy >= 0f, $"Energy: {energy}");
    }

    // ── Concurrent ComputeEnergy (read-only, max concurrency) ────────────────

    [Fact]
    public void ComputeEnergy_ConcurrentReads_AllFiniteAndNonNegative()
    {
        using var net = BuildNetwork("energy_reads");
        var x = Input(1f, 0f, 1f, 0f);
        for (int i = 0; i < 10; i++)
            net.Learn(x, OneHot(0), inferenceSteps: 5);

        var results = new float[8];

        RunThreads(8, i =>
        {
            results[i] = net.ComputeEnergy();
        });

        Assert.All(results, e => Assert.True(float.IsFinite(e) && e >= 0f, $"Energy: {e}"));
    }

    // ── Concurrent InferTrace (read-only) ────────────────────────────────────

    [Fact]
    public void InferTrace_ConcurrentCalls_AllTracesFinite()
    {
        using var net = BuildNetwork("infer_trace");
        var x = Input(0.2f, 0.8f, 0.4f, 0.6f);
        for (int i = 0; i < 10; i++)
            net.Learn(x, OneHot(1), inferenceSteps: 5);

        var traces = new float[4][];

        RunThreads(4, i =>
        {
            traces[i] = net.InferTrace(x, inferenceSteps: 20);
        });

        foreach (var trace in traces)
        {
            Assert.Equal(20, trace.Length);
            Assert.All(trace, e => Assert.True(float.IsFinite(e)));
        }
    }

    // ── Stress: many threads, mixed operations ────────────────────────────────

    [Fact]
    public void Stress_MixedOperations_NetworkSurvivesIntact()
    {
        using var net = BuildNetwork("stress");
        var x0 = Input(1f, 0f, 0f, 0f);
        var x1 = Input(0f, 1f, 0f, 0f);
        var y0  = OneHot(0);
        var y1  = OneHot(1);

        // 4 learners + 4 predictors + 2 energy readers — all concurrent
        RunThreads(10, i =>
        {
            var (x, y) = i % 2 == 0 ? (x0, y0) : (x1, y1);

            for (int iter = 0; iter < 25; iter++)
            {
                switch (i % 3)
                {
                    case 0: net.Learn(x, y, inferenceSteps: 2);         break;
                    case 1: net.Predict(x, inferenceSteps: 2);          break;
                    case 2: _ = net.ComputeEnergy();                    break;
                }
            }
        });

        float finalEnergy = net.ComputeEnergy();
        Assert.True(float.IsFinite(finalEnergy) && finalEnergy >= 0f, $"Energy: {finalEnergy}");

        float norm = net.AverageWeightNorm();
        Assert.True(float.IsFinite(norm) && norm > 0f, $"WeightNorm: {norm}");
    }

    // ── PredictWithEnergy returns consistent pair ─────────────────────────────

    [Fact]
    public void PredictWithEnergy_ConcurrentCalls_PairIsConsistent()
    {
        using var net = BuildNetwork("predict_energy");
        var x = Input(0.5f, 0.5f, 0.5f, 0.5f);
        for (int i = 0; i < 15; i++)
            net.Learn(x, OneHot(0), inferenceSteps: 5);

        var results = new (int cls, float energy)[6];

        RunThreads(6, i =>
        {
            results[i] = net.PredictWithEnergy(x, inferenceSteps: 5);
        });

        Assert.All(results, r =>
        {
            Assert.InRange(r.cls, 0, 1);
            Assert.True(float.IsFinite(r.energy) && r.energy >= 0f, $"Energy: {r.energy}");
        });
    }

    // ── Task-based (TPL) concurrent Learn + Predict ───────────────────────────

    [Fact]
    public async Task LearnAndPredict_TPL_ConcurrentTasksNoExceptions()
    {
        using var net = BuildNetwork("tpl_mixed");
        var x = Input(1f, 0f, 0f, 1f);
        var y = OneHot(0);

        var tasks = Enumerable.Range(0, 8).Select(async i =>
        {
            await Task.Yield();
            if (i % 2 == 0)
                for (int step = 0; step < 15; step++)
                    net.Learn(x, y, inferenceSteps: 3);
            else
                for (int step = 0; step < 15; step++)
                    Assert.InRange(net.Predict(x, inferenceSteps: 3), 0, 1);
        });

        await Task.WhenAll(tasks);

        float energy = net.ComputeEnergy();
        Assert.True(float.IsFinite(energy) && energy >= 0f);
    }
}
