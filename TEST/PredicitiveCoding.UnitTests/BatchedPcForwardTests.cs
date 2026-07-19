using PredictiveCoding.Core;
using PredictiveCoding.Device;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

// ROADMAP-MixedPrecision Phase 5, "batched PC forward": the GEMM-shaped batch
// settling in PredictiveCoding.Device must reproduce Network.InferGetTop
// (resetState: true) sample for sample — float tightly, mixed within Half
// tolerance — and the snapshot export must refuse dynamics it cannot batch.
public class BatchedPcForwardTests : IDisposable
{
    private readonly string _tempDir = Path.Combine(Path.GetTempPath(), $"batchedpc_{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true);
    }

    private NetworkConfig Cfg(string name, bool ffg = false, bool diag = false) => new()
    {
        Path                   = Path.Combine(_tempDir, name),
        LayerSizes             = [6, 10, 4],
        Activation             = ActivationType.Tanh,
        StateLearningRate      = 0.1f,
        WeightLearningRate     = 0.05f,
        UseFeedforwardGistInit = ffg,
        UseDiagonalPrecision   = diag,
    };

    private static DenseTensor<float>[] RandomInputs(int count, int size, int seed)
    {
        var rng = new Random(seed);
        var inputs = new DenseTensor<float>[count];
        for (int i = 0; i < count; i++)
        {
            var x = new DenseTensor<float>(size);
            for (int j = 0; j < size; j++) x.Buffer[j] = (float)(rng.NextDouble() * 2 - 1);
            inputs[i] = x;
        }
        return inputs;
    }

    [Fact]
    public void Batch_MatchesPerSampleInferGetTop()
    {
        using var net = new Network(Cfg("parity"), new Random(0));
        var x0 = DenseTensor<float>.Zeros(6);
        var x1 = DenseTensor<float>.Zeros(6); x1.Fill(1f);
        var y0 = DenseTensor<float>.Zeros(4); y0.Buffer[0] = 1f;
        var y1 = DenseTensor<float>.Zeros(4); y1.Buffer[1] = 1f;
        for (int i = 0; i < 30; i++)
        {
            net.Learn(x0, y0, inferenceSteps: 10);
            net.Learn(x1, y1, inferenceSteps: 10);
        }

        var inputs   = RandomInputs(5, 6, seed: 3);
        var snapshot = net.ExportGenerativeSnapshot();
        var batched  = BatchedPcForward.InferGetTopBatch(snapshot, inputs, inferenceSteps: 15);

        for (int i = 0; i < inputs.Length; i++)
        {
            var expected = net.InferGetTop(inputs[i], inferenceSteps: 15, resetState: true);
            for (int j = 0; j < expected.Length; j++)
            {
                float tol = 1e-3f * (1f + MathF.Abs(expected[j]));   // GEMM vs MatVec accumulation order
                Assert.True(MathF.Abs(expected[j] - batched[i][j]) <= tol,
                    $"sample {i}, unit {j}: per-sample={expected[j]}, batched={batched[i][j]}");
            }
        }
    }

    [Fact]
    public void Batch_WithFeedforwardGist_MatchesPerSample()
    {
        using var net = new Network(Cfg("parity_ffg", ffg: true), new Random(1));
        var inputs   = RandomInputs(4, 6, seed: 5);
        var snapshot = net.ExportGenerativeSnapshot();
        var batched  = BatchedPcForward.InferGetTopBatch(snapshot, inputs, inferenceSteps: 10);

        for (int i = 0; i < inputs.Length; i++)
        {
            var expected = net.InferGetTop(inputs[i], inferenceSteps: 10, resetState: true);
            for (int j = 0; j < expected.Length; j++)
            {
                float tol = 1e-3f * (1f + MathF.Abs(expected[j]));
                Assert.True(MathF.Abs(expected[j] - batched[i][j]) <= tol,
                    $"sample {i}, unit {j}: per-sample={expected[j]}, batched={batched[i][j]}");
            }
        }
    }

    [Fact]
    public void MixedBatch_TracksFloatBatch_WithinHalfTolerance()
    {
        using var net = new Network(Cfg("parity_mixed"), new Random(2));
        var inputs   = RandomInputs(4, 6, seed: 7);
        var snapshot = net.ExportGenerativeSnapshot();

        var full  = BatchedPcForward.InferGetTopBatch(snapshot, inputs, inferenceSteps: 15);
        var mixed = BatchedPcForward.InferGetTopBatch(snapshot, inputs, inferenceSteps: 15, mixed: true);

        for (int i = 0; i < inputs.Length; i++)
            for (int j = 0; j < full[i].Length; j++)
            {
                float tol = 5e-2f * (1f + MathF.Abs(full[i][j]));
                Assert.True(MathF.Abs(full[i][j] - mixed[i][j]) <= tol,
                    $"sample {i}, unit {j}: float={full[i][j]}, mixed={mixed[i][j]}");
            }
    }

    [Fact]
    public void Snapshot_RejectsDiagonalPrecision()
    {
        using var net = new Network(Cfg("diag", diag: true), new Random(4));
        Assert.Throws<NotSupportedException>(() => net.ExportGenerativeSnapshot());
    }

    [Fact]
    public void Batch_ValidatesItsArguments()
    {
        using var net = new Network(Cfg("validate"), new Random(6));
        var snapshot = net.ExportGenerativeSnapshot();

        Assert.Throws<ArgumentException>(() =>
            BatchedPcForward.InferGetTopBatch(snapshot, [], 5));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            BatchedPcForward.InferGetTopBatch(snapshot, RandomInputs(2, 6, 8), 0));
        Assert.Throws<ArgumentException>(() =>
            BatchedPcForward.InferGetTopBatch(snapshot, RandomInputs(2, 5, 9), 5));   // wrong input size
    }
}
