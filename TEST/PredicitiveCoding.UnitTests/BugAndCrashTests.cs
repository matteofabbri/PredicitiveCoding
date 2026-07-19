using PredictiveCoding.Core;
using PredictiveCoding.Model;
using PredictiveCoding.Networks;
using PredictiveCoding.Networks.MoE;
using PredictiveCoding.Networks.Temporal;

namespace PredicitiveCoding.UnitTests;

// ─────────────────────────────────────────────────────────────────────────────
// BugAndCrashTests
//
// Targeted tests hunting for bugs, crashes and deadlocks.  Each test is
// annotated with the failure mode it defends against.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class BugAndCrashTests : IDisposable
{
    private readonly string _tempDir;

    public BugAndCrashTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"pcbug_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private Network BuildNetwork(string sub, int[]? sizes = null, int seed = 0,
        bool lazy = false, bool diagPrec = false)
    {
        sizes ??= [4, 8, 2];
        var cfg = new NetworkConfig
        {
            Path                   = Path.Combine(_tempDir, sub),
            LayerSizes             = sizes,
            Activation             = ActivationType.Tanh,
            StateLearningRate      = 0.1f,
            WeightLearningRate     = 0.01f,
            LazyPersistence        = lazy,
            UseDiagonalPrecision   = diagPrec,
        };
        return new Network(cfg, new Random(seed));
    }

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    private static DenseTensor<float> OneHot(int cls, int n)
    {
        var t = DenseTensor<float>.Zeros(n);
        t.Buffer[cls] = 1f;
        return t;
    }

    private static Layer MakeHiddenLayer(int size, int lowerSize,
        ActivationType act = ActivationType.Linear)
    {
        var layer = new Layer
        {
            Index              = 1,
            Size               = size,
            LowerSize          = lowerSize,
            ActivationType     = act,
            StateLearningRate  = 0.1f,
            WeightLearningRate = 0.01f,
        };
        layer.Initialize(new Random(42));
        return layer;
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 1. LayerSerializer round-trip
    // Bug risk: wrong byte-length calculation, wrong field ordering, tensor
    //           buffer aliasing during (de)serialization.
    // ═════════════════════════════════════════════════════════════════════════

    [Fact]
    public void LayerSerializer_InputLayer_RoundTrip()
    {
        var layer = new Layer
        {
            Index = 0, Size = 4, LowerSize = 0, ActivationType = ActivationType.Tanh,
            StateLearningRate = 0.05f, WeightLearningRate = 0.02f,
        };
        layer.Initialize(new Random(1));
        layer.Mu[0] = 1.5f; layer.Mu[1] = -0.3f;
        layer.PredictionError.Fill(0.7f);

        byte[] bytes = LayerSerializer.Serialize(layer);
        var restored = LayerSerializer.AllocateAndDeserialize(bytes, 0, ActivationType.Tanh);

        Assert.Equal(layer.Size,      restored.Size);
        Assert.Equal(layer.LowerSize, restored.LowerSize);
        Assert.Null(restored.GenerativeWeights);
        Assert.Null(restored.Bias);
        Assert.Equal(1.5f,  restored.Mu[0],              precision: 5);
        Assert.Equal(-0.3f, restored.Mu[1],              precision: 5);
        Assert.Equal(0.7f,  restored.PredictionError[0], precision: 5);
    }

    [Fact]
    public void LayerSerializer_HiddenLayer_RoundTrip()
    {
        var layer = MakeHiddenLayer(size: 6, lowerSize: 4);
        layer.Mu[2] = 3.14f;
        layer.GenerativeWeights![1, 3] = -2.71f;
        layer.Bias![2] = 0.99f;

        byte[] bytes = LayerSerializer.Serialize(layer);
        var restored = LayerSerializer.AllocateAndDeserialize(bytes, 1, ActivationType.Linear);

        Assert.Equal(layer.Size,      restored.Size);
        Assert.Equal(layer.LowerSize, restored.LowerSize);
        Assert.Equal(3.14f,  restored.Mu[2],                     precision: 5);
        Assert.Equal(-2.71f, restored.GenerativeWeights![1, 3],   precision: 5);
        Assert.Equal(0.99f,  restored.Bias![2],                   precision: 5);
    }

    [Fact]
    public void LayerSerializer_WithDiagonalPrecision_RoundTrip()
    {
        var layer = MakeHiddenLayer(size: 5, lowerSize: 3);
        layer.InitDiagonalPrecision();
        layer.DiagonalPrecision![2] = 2.5f;

        byte[] bytes = LayerSerializer.Serialize(layer);
        var restored = LayerSerializer.AllocateAndDeserialize(bytes, 1, ActivationType.Linear);

        Assert.NotNull(restored.DiagonalPrecision);
        Assert.Equal(2.5f, restored.DiagonalPrecision![2], precision: 5);
    }

    [Fact]
    public void LayerSerializer_ComputeByteLength_MatchesActualSerializedSize()
    {
        // Verifies that the static calculation and actual serialization agree.
        // A mismatch would cause buffer over/underruns in PersistToStore.
        foreach (bool diag in new[] { false, true })
        foreach (bool prior in new[] { false, true })
        {
            var layer = new Layer
            {
                Index = 1, Size = 6, LowerSize = 4,
                ActivationType = ActivationType.Linear,
                StateLearningRate = 0.1f, WeightLearningRate = 0.01f,
            };
            layer.AllocateTensors(hasDiagonalPrecision: diag, hasPriorMean: prior);

            int expected = LayerSerializer.ComputeByteLength(6, 4, diag, prior);
            byte[] actual = LayerSerializer.Serialize(layer);

            Assert.Equal(expected, actual.Length);
        }
    }

    [Fact]
    public void LayerSerializer_SizeMismatch_ThrowsInvalidDataException()
    {
        var layer = MakeHiddenLayer(size: 4, lowerSize: 3);
        byte[] bytes = LayerSerializer.Serialize(layer);

        // Target has different size — must throw, not silently corrupt.
        var target = new Layer { Index = 1, Size = 8, LowerSize = 3, ActivationType = ActivationType.Linear };
        target.AllocateTensors();
        Assert.Throws<InvalidDataException>(() => LayerSerializer.Deserialize(bytes, target));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 2. Layer edge cases
    // Bug risk: division by zero in precision paths, NaN propagation,
    //           non-independent clone, temporal prior not applied.
    // ═════════════════════════════════════════════════════════════════════════

    [Fact]
    public void Layer_UpdateDiagonalPrecision_WithoutInit_Throws()
    {
        // Calling before InitDiagonalPrecision must throw, not silently NullRef.
        var layer = MakeHiddenLayer(size: 4, lowerSize: 3);
        Assert.Null(layer.DiagonalPrecision);
        Assert.Throws<InvalidOperationException>(() => layer.UpdateDiagonalPrecision(0.01f));
    }

    [Fact]
    public void Layer_UpdateDiagonalPrecision_ClampsAboveZero()
    {
        // After clamping, no element should be NaN or non-positive.
        var layer = MakeHiddenLayer(size: 4, lowerSize: 3);
        layer.InitDiagonalPrecision();
        layer.PredictionError.Fill(10f); // large error → gradient pushes precision down
        layer.UpdateDiagonalPrecision(eta: 10f);

        for (int i = 0; i < layer.Size; i++)
        {
            float pi = layer.DiagonalPrecision![i];
            Assert.True(pi >= 1e-6f,           $"Precision[{i}]={pi} below 1e-6");
            Assert.True(float.IsFinite(pi),     $"Precision[{i}]={pi} is not finite");
        }
    }

    [Fact]
    public void Layer_LocalEnergy_DiagPrecisionWithZeroes_IsFiniteAndNonNegative()
    {
        // If DiagonalPrecision[i] == 0 the energy contribution must be 0, not NaN/Inf.
        var layer = MakeHiddenLayer(size: 4, lowerSize: 3);
        layer.InitDiagonalPrecision();
        layer.DiagonalPrecision![1] = 0f; // force zero — normally clamped, but guard anyway
        layer.PredictionError.Fill(1f);

        float energy = layer.LocalEnergy();
        Assert.True(float.IsFinite(energy) && energy >= 0f, $"Energy={energy}");
    }

    [Fact]
    public void Layer_Clone_IsDeepCopy_MutationDoesNotPropagate()
    {
        var original = MakeHiddenLayer(size: 4, lowerSize: 3);
        original.Mu[0] = 1f;
        original.GenerativeWeights![0, 0] = 5f;

        var clone = original.Clone();

        // Mutate clone — original must be unaffected.
        clone.Mu[0] = 99f;
        clone.GenerativeWeights![0, 0] = 99f;

        Assert.Equal(1f, original.Mu[0]);
        Assert.Equal(5f, original.GenerativeWeights![0, 0]);
    }

    [Fact]
    public void Layer_UpdatePrecision_ClampsToMinimum()
    {
        // Large error + large eta can push precision toward 0; must clamp to 1e-6.
        var layer = MakeHiddenLayer(size: 4, lowerSize: 3);
        layer.Precision = 0.001f;
        layer.PredictionError.Fill(100f); // huge error
        layer.UpdatePrecision(eta: 1e6f);

        Assert.True(layer.Precision >= 1e-6f,       $"Precision={layer.Precision}");
        Assert.True(float.IsFinite(layer.Precision), $"Precision={layer.Precision}");
    }

    [Fact]
    public void Layer_SetPredictionError_WithPriorMean_ComputesResidual()
    {
        // Top-layer Bayesian prior: ε = Π (μ − v_p)
        var layer = new Layer
        {
            Index = 2, Size = 3, LowerSize = 0, ActivationType = ActivationType.Linear,
        };
        layer.Initialize(new Random(0));
        layer.Mu.Fill(2f);
        layer.PriorMean = Vec(1f, 1f, 1f);
        layer.Precision = 2f;

        layer.SetPredictionError(predFromAbove: null);

        // Expected: ε_i = 2 * (2 - 1) = 2
        for (int i = 0; i < 3; i++)
            Assert.Equal(2f, layer.PredictionError[i], precision: 5);
    }

    [Fact]
    public void Layer_UpdateState_WithTemporalPrior_PullsTowardPrevState()
    {
        var layer = MakeHiddenLayer(size: 4, lowerSize: 3);
        layer.Mu.Fill(1f);
        // Set a temporal prior at 0; strong temporal precision should pull mu toward 0.
        layer.TemporalPrior     = DenseTensor<float>.Zeros(4);
        layer.TemporalPrecision = 1000f;
        layer.PredictionError.Zero();
        layer.PreActivation!.Zero();
        layer.Prediction!.Zero();

        // Null lower error to isolate temporal pull.
        layer.UpdateState(lowerLayerError: null, etaState: 0.1f);

        // After a temporal pull, μ should have moved toward 0.
        for (int i = 0; i < 4; i++)
            Assert.True(layer.Mu[i] < 1f, $"Mu[{i}]={layer.Mu[i]} should have decreased toward 0");
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 3. Network input validation
    // Bug risk: silent truncation or out-of-bounds access with wrong-size inputs
    //           instead of a clear ArgumentException.
    // ═════════════════════════════════════════════════════════════════════════

    [Fact]
    public void Network_Learn_WrongInputSize_Throws()
    {
        using var net = BuildNetwork("learn_bad_input");
        var badInput = Vec(1f, 2f); // expects 4
        var label    = OneHot(0, 2);
        Assert.Throws<ArgumentException>(() => net.Learn(badInput, label, inferenceSteps: 3));
    }

    [Fact]
    public void Network_Learn_WrongLabelSize_Throws()
    {
        using var net = BuildNetwork("learn_bad_label");
        var input    = Vec(1f, 0f, 0f, 0f);
        var badLabel = OneHot(0, 5); // expects 2
        Assert.Throws<ArgumentException>(() => net.Learn(input, badLabel, inferenceSteps: 3));
    }

    [Fact]
    public void Network_Infer_WrongInputSize_Throws()
    {
        using var net = BuildNetwork("infer_bad_input");
        var badInput = Vec(1f, 2f, 3f); // expects 4
        Assert.Throws<ArgumentException>(() => net.Infer(badInput, inferenceSteps: 3));
    }

    [Fact]
    public void Network_Predict_WrongInputSize_Throws()
    {
        using var net = BuildNetwork("predict_bad_input");
        var badInput = Vec(1f); // expects 4
        Assert.Throws<ArgumentException>(() => net.Predict(badInput, inferenceSteps: 3));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 4. Network edge cases / invariants
    // Bug risk: off-by-one in inference loop, crash on 0 steps,
    //           resource leak on double dispose, NaN from lazy flush.
    // ═════════════════════════════════════════════════════════════════════════

    [Fact]
    public void Network_InferTrace_ZeroSteps_ReturnsEmptyArray()
    {
        // Must return float[0], not throw or index out of bounds.
        using var net = BuildNetwork("infer_trace_zero");
        var trace = net.InferTrace(Vec(1f, 0f, 0f, 0f), inferenceSteps: 0);
        Assert.Empty(trace);
    }

    [Fact]
    public void Network_InferTrace_MonotonicallyDecreasing_AfterTraining()
    {
        // Energy should generally decrease during inference when the network
        // has been trained; if it grows monotonically there is a sign error in
        // the state update gradient.
        using var net = BuildNetwork("infer_trace_mono");
        var x = Vec(1f, 0f, 0f, 0f);
        var y = OneHot(0, 2);
        for (int i = 0; i < 100; i++) net.Learn(x, y, inferenceSteps: 10);

        float[] trace = net.InferTrace(x, inferenceSteps: 20);

        // Compare first vs last: energy after 20 steps must be <= energy after 1.
        Assert.True(trace[19] <= trace[0],
            $"Energy should not increase after training: first={trace[0]:F4} last={trace[19]:F4}");
    }

    [Fact]
    public void Network_Dispose_CalledTwice_DoesNotThrow()
    {
        // Double-dispose must not throw — standard IDisposable contract.
        var net = BuildNetwork("double_dispose");
        net.Dispose();
        var ex = Record.Exception(() => net.Dispose());
        Assert.Null(ex);
    }

    [Fact]
    public void Network_FlushStore_LazyMode_EnergyRemainsFinite()
    {
        using var net = BuildNetwork("lazy_flush", lazy: true);
        var x = Vec(1f, 0f, 0f, 0f);
        var y = OneHot(0, 2);
        for (int i = 0; i < 10; i++) net.Learn(x, y, inferenceSteps: 5);

        net.FlushStore(); // must not throw or corrupt state

        float energy = net.ComputeEnergy();
        Assert.True(float.IsFinite(energy) && energy >= 0f, $"Energy={energy}");
    }

    [Fact]
    public void Network_DiagonalPrecision_WeightNormRemainsFinite()
    {
        // Diagonal precision learning alongside weight updates must not produce
        // NaN/Inf in weights (precision explosion → infinite gradient).
        using var net = BuildNetwork("diag_prec_norm", diagPrec: true);
        var x = Vec(1f, 0f, 0f, 0f);
        var y = OneHot(0, 2);
        for (int i = 0; i < 30; i++) net.Learn(x, y, inferenceSteps: 5);

        float norm = net.AverageWeightNorm();
        Assert.True(float.IsFinite(norm) && norm > 0f, $"WeightNorm={norm}");
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 5. Deadlock hunting
    // Bug risk: Dispose() called while worker threads hold or wait on the lock;
    //           nested lock acquisitions; RWLS not released after exception.
    // ═════════════════════════════════════════════════════════════════════════

    [Fact]
    public void Deadlock_Dispose_WhileConcurrentLearn_CompletesWithinTimeout()
    {
        // Spin up learn threads, then dispose the network while they may still
        // be running. Test fails if workers hang beyond the timeout.
        var net = BuildNetwork("deadlock_dispose");
        var x   = Vec(1f, 0f, 0f, 0f);
        var y   = OneHot(0, 2);
        var cts = new CancellationTokenSource();

        var workers = Enumerable.Range(0, 4).Select(_ => Task.Run(() =>
        {
            while (!cts.Token.IsCancellationRequested)
            {
                try { net.Learn(x, y, inferenceSteps: 2); }
                catch (ObjectDisposedException) { break; }
                catch { break; }
            }
        })).ToArray();

        Thread.Sleep(50); // let workers start
        cts.Cancel();
        net.Dispose();    // dispose while workers may still be looping

        bool completed = Task.WaitAll(workers, TimeSpan.FromSeconds(5));
        Assert.True(completed, "Workers did not complete within 5 s — possible deadlock");
    }

    [Fact]
    public void Deadlock_ConcurrentReadersAndWriter_NoDeadlock()
    {
        // Multiple InferTrace (read-lock) threads + one Learn (write-lock) thread.
        // RWLS allows multiple concurrent readers; a single writer must not starve
        // or deadlock them.  Uses a manual CancellationToken-based timeout so the
        // test can run on sync xUnit without the Timeout attribute (sync-only).
        using var net = BuildNetwork("deadlock_rw");
        var x = Vec(0.5f, 0.5f, 0.5f, 0.5f);
        var y = OneHot(1, 2);

        var errors  = new System.Collections.Concurrent.ConcurrentBag<Exception>();
        var barrier = new Barrier(6);

        var threads = Enumerable.Range(0, 6).Select(i => new Thread(() =>
        {
            barrier.SignalAndWait();
            try
            {
                for (int iter = 0; iter < 30; iter++)
                {
                    if (i == 0)
                        net.Learn(x, y, inferenceSteps: 2);
                    else
                        _ = net.InferTrace(x, inferenceSteps: 5);
                }
            }
            catch (Exception ex) { errors.Add(ex); }
        })).ToList();

        threads.ForEach(t => t.Start());
        bool allJoined = threads.All(t => t.Join(TimeSpan.FromSeconds(10)));

        Assert.True(allJoined, "Threads did not finish within 10 s — possible deadlock");
        Assert.Empty(errors);
    }

    [Fact]
    public void Deadlock_LearnTemporal_ConcurrentWithPredictTemporal_NoDeadlock()
    {
        var cfg = new NetworkConfig
        {
            Path               = Path.Combine(_tempDir, "deadlock_temporal"),
            LayerSizes         = [4, 8, 2],
            Activation         = ActivationType.Tanh,
            StateLearningRate  = 0.1f,
            WeightLearningRate = 0.01f,
        };
        // TemporalNetwork takes ownership of the Network — do not wrap baseNet in using.
        var baseNet = new Network(cfg, new Random(0));
        using var tnet = new TemporalNetwork(baseNet, temporalPrecision: 1f);

        var x = Vec(1f, 0f, 0f, 0f);
        var y = OneHot(0, 2);

        var errors  = new System.Collections.Concurrent.ConcurrentBag<Exception>();
        var barrier = new Barrier(4);

        var threads = Enumerable.Range(0, 4).Select(i => new Thread(() =>
        {
            barrier.SignalAndWait();
            try
            {
                for (int iter = 0; iter < 20; iter++)
                {
                    if (i < 2)
                        tnet.Step(x, y, inferenceSteps: 3);
                    else
                        tnet.Predict(x, inferenceSteps: 3);
                }
            }
            catch (Exception ex) { errors.Add(ex); }
        })).ToList();

        threads.ForEach(t => t.Start());
        bool allJoined = threads.All(t => t.Join(TimeSpan.FromSeconds(8)));

        Assert.True(allJoined, "Threads did not finish within 8 s — possible deadlock");
        Assert.Empty(errors);
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 6. MixtureOfExperts.ComputeGates numerical stability
    // Bug risk: NaN when all logits underflow to -Inf (softmax undefined);
    //           division by zero when sum collapses to 0.
    // ═════════════════════════════════════════════════════════════════════════

    [Fact]
    public void MoE_ComputeGates_AllZeroEnergies_UniformDistribution()
    {
        var energies = new float[] { 0f, 0f, 0f };
        var gates    = new float[3];
        MixtureOfExperts.ComputeGates(energies, gates, temperature: 1f, minGate: 0f);

        foreach (var g in gates)
        {
            Assert.True(float.IsFinite(g), $"Gate is {g}");
            Assert.Equal(1f / 3f, g, precision: 5);
        }
    }

    [Fact]
    public void MoE_ComputeGates_TemperatureZero_UniformDistribution()
    {
        // β=0: all logits are 0 → uniform softmax.
        var energies = new float[] { 10f, 1f, 100f };
        var gates    = new float[3];
        MixtureOfExperts.ComputeGates(energies, gates, temperature: 0f, minGate: 0f);

        foreach (var g in gates)
        {
            Assert.True(float.IsFinite(g), $"Gate is {g}");
            Assert.Equal(1f / 3f, g, precision: 5);
        }
    }

    [Fact]
    public void MoE_ComputeGates_ExtremeTemperature_GatesAreFinite()
    {
        // Very large β causes logits to overflow to -Inf. The stable softmax
        // should still produce finite gates (not NaN from -Inf - (-Inf)).
        var energies = new float[] { 1f, 2f };
        var gates    = new float[2];
        MixtureOfExperts.ComputeGates(energies, gates, temperature: float.MaxValue, minGate: 0f);

        Assert.All(gates, g => Assert.True(float.IsFinite(g), $"Gate={g}"));
        Assert.Equal(1f, gates.Sum(), precision: 4);
    }

    [Fact]
    public void MoE_ComputeGates_MinGateFloor_SumIsOne()
    {
        // Applying a min-gate floor and renormalising must still yield a proper
        // probability distribution summing to 1.
        var energies = new float[] { 1f, 1f, 1f, 1f };
        var gates    = new float[4];
        MixtureOfExperts.ComputeGates(energies, gates, temperature: 1f, minGate: 0.1f);

        Assert.Equal(1f, gates.Sum(), precision: 4);
        Assert.All(gates, g =>
        {
            Assert.True(float.IsFinite(g), $"Gate={g}");
            Assert.True(g >= 0f, $"Negative gate={g}");
        });
    }

    [Fact]
    public void MoE_PredictDetailed_AfterLearning_EnergiesAreFinite()
    {
        var basePath = Path.Combine(_tempDir, "moe_detailed");
        var cfg = new MixtureOfExpertsConfig
        {
            BasePath          = basePath,
            ExpertCount       = 3,
            ExpertConfig      = new NetworkConfig
            {
                LayerSizes         = [4, 8, 2],
                Activation         = ActivationType.Tanh,
                StateLearningRate  = 0.1f,
                WeightLearningRate = 0.01f,
            },
            GatingTemperature = 1f,
            MinGateWeight     = 0f,
        };
        using var moe = new MixtureOfExperts(cfg, new Random(0));

        var x = Vec(1f, 0f, 0f, 0f);
        var y = OneHot(0, 2);
        for (int i = 0; i < 20; i++) moe.Learn(x, y, inferenceSteps: 5);

        var result = moe.PredictDetailed(x, inferenceSteps: 5);

        Assert.InRange(result.PredictedClass, 0, 1);
        Assert.InRange(result.WinningExpert,  0, 2);
        Assert.All(result.ExpertEnergies, e =>
            Assert.True(float.IsFinite(e) && e >= 0f, $"Energy={e}"));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 7. TemporalNetwork edge cases
    // Bug risk: wrong prior array length causes ArgumentException deep inside
    //           InjectTemporalPriors; Reset must clear all prior state.
    // ═════════════════════════════════════════════════════════════════════════

    [Fact]
    public void TemporalNetwork_Reset_ClearsPriorState()
    {
        var cfg = new NetworkConfig
        {
            Path               = Path.Combine(_tempDir, "temporal_reset"),
            LayerSizes         = [4, 8, 2],
            Activation         = ActivationType.Tanh,
            StateLearningRate  = 0.1f,
            WeightLearningRate = 0.01f,
        };
        // TemporalNetwork owns the underlying Network — no separate using for baseNet.
        using var tnet = new TemporalNetwork(new Network(cfg, new Random(0)), temporalPrecision: 1f);

        var x = Vec(1f, 0f, 0f, 0f);
        var y = OneHot(0, 2);

        tnet.Step(x, y, inferenceSteps: 5);
        Assert.True(tnet.HasPrior);

        tnet.Reset();
        Assert.False(tnet.HasPrior);
    }

    [Fact]
    public void TemporalNetwork_PredictBeforeStep_DoesNotCrash()
    {
        // HasPrior=false on first call; InjectTemporalPriors is skipped with null priors.
        var cfg = new NetworkConfig
        {
            Path               = Path.Combine(_tempDir, "temporal_predict_cold"),
            LayerSizes         = [4, 8, 2],
            Activation         = ActivationType.Tanh,
            StateLearningRate  = 0.1f,
            WeightLearningRate = 0.01f,
        };
        using var tnet = new TemporalNetwork(new Network(cfg, new Random(0)), temporalPrecision: 1f);

        var x = Vec(0.5f, 0.1f, 0.9f, 0.3f);
        int cls = tnet.Predict(x, inferenceSteps: 5);
        Assert.InRange(cls, 0, 1);
    }

    [Fact]
    public void TemporalNetwork_MultipleStepsAndReset_EnergyRemainsFinite()
    {
        var cfg = new NetworkConfig
        {
            Path               = Path.Combine(_tempDir, "temporal_energy"),
            LayerSizes         = [4, 8, 2],
            Activation         = ActivationType.Tanh,
            StateLearningRate  = 0.1f,
            WeightLearningRate = 0.01f,
        };
        using var tnet = new TemporalNetwork(new Network(cfg, new Random(0)), temporalPrecision: 1f);

        var x = Vec(1f, 0f, 0f, 0f);
        var y = OneHot(0, 2);

        for (int seq = 0; seq < 3; seq++)
        {
            for (int t = 0; t < 5; t++)
                tnet.Step(x, y, inferenceSteps: 5);
            tnet.Reset();
        }

        float energy = tnet.BaseNetwork.ComputeEnergy();
        Assert.True(float.IsFinite(energy) && energy >= 0f, $"Energy={energy}");
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 8. NetworkConfig validation
    // Bug risk: invalid configs reach the constructor and crash later during
    //           inference rather than at construction time.
    // ═════════════════════════════════════════════════════════════════════════

    [Fact]
    public void NetworkConfig_SingleLayer_Throws()
    {
        var cfg = new NetworkConfig
        {
            Path       = Path.Combine(_tempDir, "bad_layers"),
            LayerSizes = [4],
            Activation = ActivationType.Linear,
        };
        Assert.Throws<InvalidOperationException>(() => new Network(cfg, new Random(0)));
    }

    [Fact]
    public void NetworkConfig_NegativeStateLr_Throws()
    {
        var cfg = new NetworkConfig
        {
            Path              = Path.Combine(_tempDir, "bad_lr"),
            LayerSizes        = [4, 2],
            StateLearningRate = -0.1f,
        };
        Assert.Throws<InvalidOperationException>(() => new Network(cfg, new Random(0)));
    }

    [Fact]
    public void NetworkConfig_InputShapeMismatch_Throws()
    {
        var cfg = new NetworkConfig
        {
            Path        = Path.Combine(_tempDir, "bad_shape"),
            LayerSizes  = [4, 2],
            InputShape  = [3, 2], // 3*2=6 != 4
        };
        Assert.Throws<InvalidOperationException>(() => new Network(cfg, new Random(0)));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 9. WeightDecay self-axpy correctness
    // Bug risk: InPlaceAxpy(alpha, W, W) — source and destination alias the
    //           same buffer.  Must produce W*(1+alpha) = W*(1-lr*decay),
    //           not corrupt the buffer.
    // ═════════════════════════════════════════════════════════════════════════

    [Fact]
    public void Layer_UpdateWeights_WithDecay_WeightNormDecreases()
    {
        // After a pure decay step (zero prediction error so the Hebbian term
        // is 0) the weight norm must strictly decrease.
        var layer = MakeHiddenLayer(size: 4, lowerSize: 3);
        layer.WeightDecay = 0.1f;

        // Zero prediction error → ΔW = 0; decay shrinks W.
        var zeroDelta = DenseTensor<float>.Zeros(3);
        float normBefore = layer.WeightNorm();
        layer.UpdateWeights(zeroDelta, etaWeight: 0.01f);
        float normAfter = layer.WeightNorm();

        Assert.True(normAfter < normBefore,
            $"Weight decay should shrink norm: before={normBefore:F4} after={normAfter:F4}");
        Assert.True(float.IsFinite(normAfter) && normAfter >= 0f,
            $"Weight norm is not finite: {normAfter}");
    }
}
