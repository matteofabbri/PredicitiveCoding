using PredictiveCoding.Core;
using PredictiveCoding.Model;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Tests for Network with large input and output dimensions.
///
/// Scenarios covered:
///   1D-Large : flat vector  [1024]        → 512 → 256 → 64
///   2D-Large : image        [32, 32]      → 512 → 20
///   3D-Large : volume       [8, 16, 16]   → 512 → 32
///   Wide-out : flat vector  [256]         → 512 → 128 (many output classes)
///   Deep     : flat vector  [128]         → 256 → 128 → 64 → 32 → 16
/// </summary>
public class NetworkLargeDimsTests : IDisposable
{
    private readonly string _tempDir;

    public NetworkLargeDimsTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"pclarge_{Guid.NewGuid():N}");
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }

    // ── Config helpers ────────────────────────────────────────────────────────

    private NetworkConfig Cfg(string sub, int[] layerSizes,
        int[]? inputShape = null, int[]? outputShape = null) => new()
    {
        Path               = Path.Combine(_tempDir, sub),
        LayerSizes         = layerSizes,
        Activation         = ActivationType.Tanh,
        StateLearningRate  = 0.1f,
        WeightLearningRate = 0.01f,
        InputShape         = inputShape,
        OutputShape        = outputShape,
    };

    // ── Input helpers ─────────────────────────────────────────────────────────

    private static DenseTensor<float> FlatInput(int size, float value = 0.5f)
    {
        var t = new DenseTensor<float>(size);
        t.Fill(value);
        return t;
    }

    private static DenseTensor<float> FlatInputRandom(int size, Random rng)
    {
        var t = new DenseTensor<float>(size);
        for (int i = 0; i < size; i++) t[i] = (float)rng.NextDouble();
        return t;
    }

    private static DenseTensor<float> Image2D(int rows, int cols, float value)
    {
        var t = new DenseTensor<float>(rows, cols);
        t.Fill(value);
        return t;
    }

    private static DenseTensor<float> CheckerImage2D(int rows, int cols)
    {
        var t = new DenseTensor<float>(rows, cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                t[r, c] = (r + c) % 2 == 0 ? 1f : 0f;
        return t;
    }

    private static DenseTensor<float> Volume3D(int d0, int d1, int d2, float value)
    {
        var t = new DenseTensor<float>(d0, d1, d2);
        t.Fill(value);
        return t;
    }

    private static DenseTensor<float> CheckerVolume3D(int d0, int d1, int d2)
    {
        var t = new DenseTensor<float>(d0, d1, d2);
        for (int i = 0; i < d0; i++)
            for (int j = 0; j < d1; j++)
                for (int k = 0; k < d2; k++)
                    t[i, j, k] = (i + j + k) % 2 == 0 ? 1f : 0f;
        return t;
    }

    private static DenseTensor<float> OneHot(int index, int size)
    {
        var t = DenseTensor<float>.Zeros(size);
        t.Buffer[index] = 1f;
        return t;
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 1. Large 1-D input / small output  [1024 → 512 → 256 → 64]
    // ═════════════════════════════════════════════════════════════════════════

    private NetworkConfig Large1DCfg(string sub = "1d") =>
        Cfg(sub, [1024, 512, 256, 64]);

    [Fact]
    public void Large1D_Construction_DoesNotThrow()
    {
        using var net = new Network(Large1DCfg(), new Random(0));
        Assert.Equal(4, net.LayerCount);
    }

    [Fact]
    public void Large1D_AverageWeightNorm_IsPositiveAndFinite()
    {
        using var net = new Network(Large1DCfg(), new Random(0));
        float norm = net.AverageWeightNorm();
        Assert.True(norm > 0f && float.IsFinite(norm));
    }

    [Fact]
    public void Large1D_Infer_DoesNotThrow()
    {
        using var net = new Network(Large1DCfg(), new Random(0));
        net.Infer(FlatInput(1024), inferenceSteps: 3);
    }

    [Fact]
    public void Large1D_Infer_IsTransient_WeightsUnchanged()
    {
        using var net = new Network(Large1DCfg(), new Random(0));
        float normBefore = net.AverageWeightNorm();
        net.Infer(FlatInputRandom(1024, new Random(1)), inferenceSteps: 5);
        Assert.Equal(normBefore, net.AverageWeightNorm(), precision: 6);
    }

    [Fact]
    public void Large1D_Infer_EnergyFiniteAndNonNegative()
    {
        using var net = new Network(Large1DCfg(), new Random(0));
        net.Infer(FlatInput(1024, 1f), inferenceSteps: 5);
        float energy = net.ComputeEnergy();
        Assert.True(energy >= 0f && float.IsFinite(energy));
    }

    [Fact]
    public void Large1D_Learn_ChangesWeights()
    {
        using var net = new Network(Large1DCfg(), new Random(0));
        float normBefore = net.AverageWeightNorm();
        net.Learn(FlatInput(1024), OneHot(0, 64), inferenceSteps: 3);
        Assert.NotEqual(normBefore, net.AverageWeightNorm());
    }

    [Fact]
    public void Large1D_Learn_EnergyRemainsFinite()
    {
        using var net = new Network(Large1DCfg(), new Random(0));
        var x = FlatInput(1024, 0.3f);
        var y = OneHot(7, 64);
        for (int i = 0; i < 10; i++)
            net.Learn(x, y, inferenceSteps: 5);
        Assert.True(float.IsFinite(net.ComputeEnergy()));
    }

    [Fact]
    public void Large1D_Predict_ReturnsValidClassIndex()
    {
        using var net = new Network(Large1DCfg(), new Random(0));
        int cls = net.Predict(FlatInput(1024), inferenceSteps: 3);
        Assert.InRange(cls, 0, 63);
    }

    [Fact]
    public void Large1D_Predict_IsDeterministic()
    {
        using var net = new Network(Large1DCfg(), new Random(0));
        var x = FlatInputRandom(1024, new Random(99));
        Assert.Equal(net.Predict(x, inferenceSteps: 5), net.Predict(x, inferenceSteps: 5));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 2. Large 2-D input  [32×32] → 512 → 20
    // ═════════════════════════════════════════════════════════════════════════

    private const int ImgR = 32, ImgC = 32, NumClasses2D = 20;

    private NetworkConfig Large2DCfg(string sub = "2d") =>
        Cfg(sub, [ImgR * ImgC, 512, NumClasses2D],
            inputShape: [ImgR, ImgC]);

    [Fact]
    public void Large2D_Infer_DoesNotThrow()
    {
        using var net = new Network(Large2DCfg(), new Random(0));
        net.Infer(Image2D(ImgR, ImgC, 0.5f), inferenceSteps: 3);
    }

    [Fact]
    public void Large2D_Infer_IsTransient_WeightsUnchanged()
    {
        using var net = new Network(Large2DCfg(), new Random(0));
        float normBefore = net.AverageWeightNorm();
        net.Infer(CheckerImage2D(ImgR, ImgC), inferenceSteps: 5);
        Assert.Equal(normBefore, net.AverageWeightNorm(), precision: 6);
    }

    [Fact]
    public void Large2D_Learn_ChangesWeights()
    {
        using var net = new Network(Large2DCfg(), new Random(0));
        float normBefore = net.AverageWeightNorm();
        net.Learn(Image2D(ImgR, ImgC, 0f), OneHot(0, NumClasses2D), inferenceSteps: 3);
        Assert.NotEqual(normBefore, net.AverageWeightNorm());
    }

    [Fact]
    public void Large2D_Learn_EnergyRemainsFinite()
    {
        using var net = new Network(Large2DCfg(), new Random(0));
        var x = CheckerImage2D(ImgR, ImgC);
        var y = OneHot(3, NumClasses2D);
        for (int i = 0; i < 10; i++)
            net.Learn(x, y, inferenceSteps: 5);
        Assert.True(float.IsFinite(net.ComputeEnergy()));
    }

    [Fact]
    public void Large2D_Predict_ReturnsValidClassIndex()
    {
        using var net = new Network(Large2DCfg(), new Random(0));
        int cls = net.Predict(Image2D(ImgR, ImgC, 1f), inferenceSteps: 3);
        Assert.InRange(cls, 0, NumClasses2D - 1);
    }

    [Fact]
    public void Large2D_Predict_IsDeterministic()
    {
        using var net = new Network(Large2DCfg(), new Random(0));
        var x = CheckerImage2D(ImgR, ImgC);
        Assert.Equal(net.Predict(x, inferenceSteps: 5), net.Predict(x, inferenceSteps: 5));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 3. Large 3-D input  [8×16×16] → 512 → 32
    // ═════════════════════════════════════════════════════════════════════════

    private const int VD = 8, VH = 16, VW = 16, NumClasses3D = 32;

    private NetworkConfig Large3DCfg(string sub = "3d") =>
        Cfg(sub, [VD * VH * VW, 512, NumClasses3D],
            inputShape: [VD, VH, VW]);

    [Fact]
    public void Large3D_Infer_DoesNotThrow()
    {
        using var net = new Network(Large3DCfg(), new Random(0));
        net.Infer(Volume3D(VD, VH, VW, 0.5f), inferenceSteps: 3);
    }

    [Fact]
    public void Large3D_Infer_IsTransient_WeightsUnchanged()
    {
        using var net = new Network(Large3DCfg(), new Random(0));
        float normBefore = net.AverageWeightNorm();
        net.Infer(CheckerVolume3D(VD, VH, VW), inferenceSteps: 5);
        Assert.Equal(normBefore, net.AverageWeightNorm(), precision: 6);
    }

    [Fact]
    public void Large3D_Learn_ChangesWeights()
    {
        using var net = new Network(Large3DCfg(), new Random(0));
        float normBefore = net.AverageWeightNorm();
        net.Learn(Volume3D(VD, VH, VW, 0f), OneHot(0, NumClasses3D), inferenceSteps: 3);
        Assert.NotEqual(normBefore, net.AverageWeightNorm());
    }

    [Fact]
    public void Large3D_Learn_EnergyRemainsFinite()
    {
        using var net = new Network(Large3DCfg(), new Random(0));
        var x = CheckerVolume3D(VD, VH, VW);
        var y = OneHot(5, NumClasses3D);
        for (int i = 0; i < 10; i++)
            net.Learn(x, y, inferenceSteps: 5);
        Assert.True(float.IsFinite(net.ComputeEnergy()));
    }

    [Fact]
    public void Large3D_Predict_ReturnsValidClassIndex()
    {
        using var net = new Network(Large3DCfg(), new Random(0));
        int cls = net.Predict(Volume3D(VD, VH, VW, 1f), inferenceSteps: 3);
        Assert.InRange(cls, 0, NumClasses3D - 1);
    }

    [Fact]
    public void Large3D_Predict_IsDeterministic()
    {
        using var net = new Network(Large3DCfg(), new Random(0));
        var x = CheckerVolume3D(VD, VH, VW);
        Assert.Equal(net.Predict(x, inferenceSteps: 5), net.Predict(x, inferenceSteps: 5));
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 4. Wide output  [256] → 512 → 128 classes
    // ═════════════════════════════════════════════════════════════════════════

    private const int WideIn = 256, WideOut = 128;

    private NetworkConfig WideOutCfg(string sub = "wide") =>
        Cfg(sub, [WideIn, 512, WideOut]);

    [Fact]
    public void WideOut_Predict_ReturnsValidClassIndex()
    {
        using var net = new Network(WideOutCfg(), new Random(0));
        int cls = net.Predict(FlatInput(WideIn), inferenceSteps: 3);
        Assert.InRange(cls, 0, WideOut - 1);
    }

    [Fact]
    public void WideOut_ComputeLayerEnergies_AllFiniteAndNonNegative()
    {
        using var net = new Network(WideOutCfg(), new Random(0));
        net.Infer(FlatInput(WideIn, 0.8f), inferenceSteps: 5);
        Assert.All(net.ComputeLayerEnergies(), e => Assert.True(e >= 0f && float.IsFinite(e)));
    }

    [Fact]
    public void WideOut_Learn_ChangesWeights()
    {
        using var net = new Network(WideOutCfg(), new Random(0));
        float normBefore = net.AverageWeightNorm();
        net.Learn(FlatInput(WideIn, 0.5f), OneHot(64, WideOut), inferenceSteps: 3);
        Assert.NotEqual(normBefore, net.AverageWeightNorm());
    }

    // ═════════════════════════════════════════════════════════════════════════
    // 5. Deep network  [128] → 256 → 128 → 64 → 32 → 16
    // ═════════════════════════════════════════════════════════════════════════

    private const int DeepIn = 128, DeepOut = 16;

    private NetworkConfig DeepCfg(string sub = "deep") =>
        Cfg(sub, [DeepIn, 256, 128, 64, 32, DeepOut]);

    [Fact]
    public void Deep_LayerCount_IsCorrect()
    {
        using var net = new Network(DeepCfg(), new Random(0));
        Assert.Equal(6, net.LayerCount);
    }

    [Fact]
    public void Deep_Infer_DoesNotThrow()
    {
        using var net = new Network(DeepCfg(), new Random(0));
        net.Infer(FlatInput(DeepIn), inferenceSteps: 3);
    }

    [Fact]
    public void Deep_Infer_IsTransient_WeightsUnchanged()
    {
        using var net = new Network(DeepCfg(), new Random(0));
        float normBefore = net.AverageWeightNorm();
        net.Infer(FlatInputRandom(DeepIn, new Random(7)), inferenceSteps: 5);
        Assert.Equal(normBefore, net.AverageWeightNorm(), precision: 6);
    }

    [Fact]
    public void Deep_Learn_ChangesWeights()
    {
        using var net = new Network(DeepCfg(), new Random(0));
        float normBefore = net.AverageWeightNorm();
        net.Learn(FlatInput(DeepIn, 0.5f), OneHot(0, DeepOut), inferenceSteps: 3);
        Assert.NotEqual(normBefore, net.AverageWeightNorm());
    }

    [Fact]
    public void Deep_Learn_EnergyRemainsFinite()
    {
        using var net = new Network(DeepCfg(), new Random(0));
        var x = FlatInput(DeepIn, 0.5f);
        var y = OneHot(2, DeepOut);
        for (int i = 0; i < 10; i++)
            net.Learn(x, y, inferenceSteps: 5);
        Assert.True(float.IsFinite(net.ComputeEnergy()));
    }

    [Fact]
    public void Deep_ComputeLayerEnergies_LengthIsLayerCountMinusOne()
    {
        using var net = new Network(DeepCfg(), new Random(0));
        Assert.Equal(net.LayerCount - 1, net.ComputeLayerEnergies().Length);
    }

    [Fact]
    public void Deep_Predict_ReturnsValidClassIndex()
    {
        using var net = new Network(DeepCfg(), new Random(0));
        int cls = net.Predict(FlatInput(DeepIn), inferenceSteps: 3);
        Assert.InRange(cls, 0, DeepOut - 1);
    }

    [Fact]
    public void Deep_Predict_IsDeterministic()
    {
        using var net = new Network(DeepCfg(), new Random(0));
        var x = FlatInputRandom(DeepIn, new Random(13));
        Assert.Equal(net.Predict(x, inferenceSteps: 5), net.Predict(x, inferenceSteps: 5));
    }
}
