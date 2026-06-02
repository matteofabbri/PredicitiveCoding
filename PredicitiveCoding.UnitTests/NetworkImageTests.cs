using PredictiveCoding.Core;
using PredictiveCoding.Model;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Tests for Network with 2-D input (InputShape) and 2-D label (OutputShape).
///
/// Layout used throughout:
///   image  : [4, 4] → 16 pixels  (LayerSizes[0] = 16)
///   label  : [2, 2] → 4 classes  (LayerSizes[^1] = 4)
/// </summary>
public class NetworkImageTests : IDisposable
{
    // 4×4 image → hidden → 2×2 label map
    private const int Rows   = 4;
    private const int Cols   = 4;
    private const int LRows  = 2;
    private const int LCols  = 2;

    private readonly string _tempDir;

    public NetworkImageTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"pcimg_{Guid.NewGuid():N}");
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
        LayerSizes         = [Rows * Cols, 32, LRows * LCols],
        Activation         = ActivationType.Tanh,
        StateLearningRate  = 0.1f,
        WeightLearningRate = 0.01f,
        InputShape         = [Rows, Cols],
        OutputShape        = [LRows, LCols],
    };

    /// <summary>Creates a [Rows, Cols] image filled with <paramref name="value"/>.</summary>
    private static DenseTensor<float> SolidImage(float value)
    {
        var t = new DenseTensor<float>(Rows, Cols);
        t.Fill(value);
        return t;
    }

    /// <summary>Creates a [Rows, Cols] checkerboard image.</summary>
    private static DenseTensor<float> CheckerImage()
    {
        var t = new DenseTensor<float>(Rows, Cols);
        for (int r = 0; r < Rows; r++)
            for (int c = 0; c < Cols; c++)
                t[r, c] = (r + c) % 2 == 0 ? 1f : 0f;
        return t;
    }

    /// <summary>One-hot label in [LRows, LCols] layout.</summary>
    private static DenseTensor<float> Label2D(int index)
    {
        var t = new DenseTensor<float>(LRows, LCols);
        t.Buffer[index] = 1f;
        return t;
    }

    // ── Config validation ─────────────────────────────────────────────────────

    [Fact]
    public void Validate_InputShapeProductMismatch_Throws()
    {
        var cfg = MakeCfg() with { InputShape = [3, 3] }; // 9 ≠ 16
        Assert.Throws<InvalidOperationException>(() => new Network(cfg, new Random(0)));
    }

    [Fact]
    public void Validate_OutputShapeProductMismatch_Throws()
    {
        var cfg = MakeCfg() with { OutputShape = [3, 3] }; // 9 ≠ 4
        Assert.Throws<InvalidOperationException>(() => new Network(cfg, new Random(0)));
    }

    [Fact]
    public void Validate_ZeroDimension_Throws()
    {
        var cfg = MakeCfg() with { InputShape = [0, Cols] };
        Assert.Throws<InvalidOperationException>(() => new Network(cfg, new Random(0)));
    }

    // ── Infer ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Infer_2DImage_DoesNotThrow()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        net.Infer(SolidImage(0.5f), inferenceSteps: 3);
    }

    [Fact]
    public void Infer_WrongInputShape_Throws()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        // [3, 3] has 9 elements but network expects 16.
        var wrong = new DenseTensor<float>(3, 3);
        Assert.Throws<ArgumentException>(() => net.Infer(wrong, inferenceSteps: 1));
    }

    [Fact]
    public void Infer_2DImage_IsTransient_WeightsUnchanged()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        float normBefore = net.AverageWeightNorm();

        net.Infer(CheckerImage(), inferenceSteps: 5);

        Assert.Equal(normBefore, net.AverageWeightNorm(), precision: 6);
    }

    // ── Learn ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Learn_2DImageAnd2DLabel_DoesNotThrow()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        net.Learn(SolidImage(0f), Label2D(0), inferenceSteps: 3);
    }

    [Fact]
    public void Learn_WrongInputShape_Throws()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        var wrong = new DenseTensor<float>(3, 3);
        Assert.Throws<ArgumentException>(() =>
            net.Learn(wrong, Label2D(0), inferenceSteps: 1));
    }

    [Fact]
    public void Learn_WrongLabelShape_Throws()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        var wrongLabel = new DenseTensor<float>(3, 3); // 9 ≠ 4
        Assert.Throws<ArgumentException>(() =>
            net.Learn(SolidImage(0f), wrongLabel, inferenceSteps: 1));
    }

    [Fact]
    public void Learn_2DInputAndLabel_ChangesWeights()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        float normBefore = net.AverageWeightNorm();

        net.Learn(CheckerImage(), Label2D(1), inferenceSteps: 5);

        Assert.NotEqual(normBefore, net.AverageWeightNorm());
    }

    // ── Predict ───────────────────────────────────────────────────────────────

    [Fact]
    public void Predict_2DImage_ReturnsValidClassIndex()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        int numClasses = LRows * LCols;

        int predicted = net.Predict(SolidImage(1f), inferenceSteps: 5);

        Assert.InRange(predicted, 0, numClasses - 1);
    }

    [Fact]
    public void Predict_2DImage_IsDeterministic()
    {
        using var net = new Network(MakeCfg(), new Random(42));
        var image = CheckerImage();

        int first  = net.Predict(image, inferenceSteps: 10);
        int second = net.Predict(image, inferenceSteps: 10);

        Assert.Equal(first, second);
    }

    // ── Learn + Predict ───────────────────────────────────────────────────────

    [Fact]
    public void LearnRepeated_2DPatterns_ConvergesOnTwoClasses()
    {
        // Solid-black → class 0 ([1,0,0,0] in the 2×2 label map)
        // Checker     → class 1 ([0,1,0,0])
        var cfg = MakeCfg("converge") with
        {
            LayerSizes         = [Rows * Cols, 64, LRows * LCols],
            StateLearningRate  = 0.2f,
            WeightLearningRate = 0.05f,
        };
        using var net = new Network(cfg, new Random(0));

        var black   = SolidImage(0f);
        var checker = CheckerImage();
        var y0      = Label2D(0);
        var y1      = Label2D(1);

        for (int i = 0; i < 300; i++)
        {
            net.Learn(black,   y0, inferenceSteps: 10);
            net.Learn(checker, y1, inferenceSteps: 10);
        }

        Assert.Equal(0, net.Predict(black,   inferenceSteps: 20));
        Assert.Equal(1, net.Predict(checker, inferenceSteps: 20));
    }
}
