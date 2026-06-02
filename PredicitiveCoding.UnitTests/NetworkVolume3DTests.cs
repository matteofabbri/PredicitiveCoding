using PredictiveCoding.Core;
using PredictiveCoding.Model;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Tests for Network with 3-D input (InputShape) and 3-D label (OutputShape).
///
/// Layout used throughout:
///   input : [3, 4, 4]  → 48 voxels  (channels × height × width, LayerSizes[0] = 48)
///   output: [2, 2, 2]  → 8 classes  (depth × rows × cols,       LayerSizes[^1] = 8)
/// </summary>
public class NetworkVolume3DTests : IDisposable
{
    private const int C  = 3, H  = 4, W  = 4;   // input  shape
    private const int OD = 2, OR = 2, OC = 2;    // output shape

    private readonly string _tempDir;

    public NetworkVolume3DTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"pcvol3d_{Guid.NewGuid():N}");
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    private NetworkConfig MakeCfg(string subDir = "net") => new()
    {
        Path               = Path.Combine(_tempDir, subDir),
        LayerSizes         = [C * H * W, 64, OD * OR * OC],
        Activation         = ActivationType.Tanh,
        StateLearningRate  = 0.1f,
        WeightLearningRate = 0.01f,
        InputShape         = [C, H, W],
        OutputShape        = [OD, OR, OC],
    };

    /// <summary>Creates a [C, H, W] volume filled with <paramref name="value"/>.</summary>
    private static DenseTensor<float> SolidVolume(float value)
    {
        var t = new DenseTensor<float>(C, H, W);
        t.Fill(value);
        return t;
    }

    /// <summary>
    /// Creates a [C, H, W] volume where voxel values alternate 0/1
    /// based on (channel + row + col) parity.
    /// </summary>
    private static DenseTensor<float> CheckerVolume()
    {
        var t = new DenseTensor<float>(C, H, W);
        for (int c = 0; c < C; c++)
            for (int r = 0; r < H; r++)
                for (int w = 0; w < W; w++)
                    t[c, r, w] = (c + r + w) % 2 == 0 ? 1f : 0f;
        return t;
    }

    /// <summary>One-hot label in [OD, OR, OC] layout.</summary>
    private static DenseTensor<float> Label3D(int index)
    {
        var t = new DenseTensor<float>(OD, OR, OC);
        t.Buffer[index] = 1f;
        return t;
    }

    // ── Config validation ─────────────────────────────────────────────────────

    [Fact]
    public void Validate_3DInputShapeProductMismatch_Throws()
    {
        var cfg = MakeCfg() with { InputShape = [2, 3, 3] }; // 18 ≠ 48
        Assert.Throws<InvalidOperationException>(() => new Network(cfg, new Random(0)));
    }

    [Fact]
    public void Validate_3DOutputShapeProductMismatch_Throws()
    {
        var cfg = MakeCfg() with { OutputShape = [2, 3, 3] }; // 18 ≠ 8
        Assert.Throws<InvalidOperationException>(() => new Network(cfg, new Random(0)));
    }

    [Fact]
    public void Validate_3DInputShapeWithZeroDimension_Throws()
    {
        var cfg = MakeCfg() with { InputShape = [0, H, W] };
        Assert.Throws<InvalidOperationException>(() => new Network(cfg, new Random(0)));
    }

    [Fact]
    public void Validate_3DOutputShapeWithZeroDimension_Throws()
    {
        var cfg = MakeCfg() with { OutputShape = [0, OR, OC] };
        Assert.Throws<InvalidOperationException>(() => new Network(cfg, new Random(0)));
    }

    // ── Infer ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Infer_3DVolume_DoesNotThrow()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        net.Infer(SolidVolume(0.5f), inferenceSteps: 3);
    }

    [Fact]
    public void Infer_3DVolume_IsTransient_WeightsUnchanged()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        float normBefore = net.AverageWeightNorm();

        net.Infer(CheckerVolume(), inferenceSteps: 5);

        Assert.Equal(normBefore, net.AverageWeightNorm(), precision: 6);
    }

    [Fact]
    public void Infer_Wrong3DInputShape_Throws()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        // [2, 3, 4] = 24 ≠ 48
        var wrong = new DenseTensor<float>(2, 3, 4);
        Assert.Throws<ArgumentException>(() => net.Infer(wrong, inferenceSteps: 1));
    }

    // ── Learn ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Learn_3DVolumeAnd3DLabel_DoesNotThrow()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        net.Learn(SolidVolume(0f), Label3D(0), inferenceSteps: 3);
    }

    [Fact]
    public void Learn_3DVolumeAnd3DLabel_ChangesWeights()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        float normBefore = net.AverageWeightNorm();

        net.Learn(CheckerVolume(), Label3D(1), inferenceSteps: 5);

        Assert.NotEqual(normBefore, net.AverageWeightNorm());
    }

    [Fact]
    public void Learn_Wrong3DInputShape_Throws()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        var wrong = new DenseTensor<float>(2, 3, 4); // 24 ≠ 48
        Assert.Throws<ArgumentException>(() =>
            net.Learn(wrong, Label3D(0), inferenceSteps: 1));
    }

    [Fact]
    public void Learn_Wrong3DLabelShape_Throws()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        var wrongLabel = new DenseTensor<float>(2, 2, 3); // 12 ≠ 8
        Assert.Throws<ArgumentException>(() =>
            net.Learn(SolidVolume(0f), wrongLabel, inferenceSteps: 1));
    }

    // ── Predict ───────────────────────────────────────────────────────────────

    [Fact]
    public void Predict_3DVolume_ReturnsValidClassIndex()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        int numClasses = OD * OR * OC;

        int predicted = net.Predict(SolidVolume(1f), inferenceSteps: 5);

        Assert.InRange(predicted, 0, numClasses - 1);
    }

    [Fact]
    public void Predict_3DVolume_IsDeterministic()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        var volume = CheckerVolume();

        int first  = net.Predict(volume, inferenceSteps: 10);
        int second = net.Predict(volume, inferenceSteps: 10);

        Assert.Equal(first, second);
    }

    // ── Learn + Predict ───────────────────────────────────────────────────────

    [Fact]
    public void LearnRepeated_3DPatterns_ConvergesOnTwoClasses()
    {
        // Solid-black → class 0 / Checker → class 1 (trivially separable)
        var cfg = MakeCfg("converge") with
        {
            LayerSizes         = [C * H * W, 128, OD * OR * OC],
            StateLearningRate  = 0.2f,
            WeightLearningRate = 0.05f,
        };
        using var net = new Network(cfg, new Random(0));

        var black   = SolidVolume(0f);
        var checker = CheckerVolume();
        var y0      = Label3D(0);
        var y1      = Label3D(1);

        for (int i = 0; i < 300; i++)
        {
            net.Learn(black,   y0, inferenceSteps: 10);
            net.Learn(checker, y1, inferenceSteps: 10);
        }

        Assert.Equal(0, net.Predict(black,   inferenceSteps: 20));
        Assert.Equal(1, net.Predict(checker, inferenceSteps: 20));
    }
}
