using PredictiveCoding.Core;
using PredictiveCoding.Model;

namespace PredicitiveCoding.UnitTests;

public class NetworkTests
{
    private static Network MakeNet(int[] sizes, ActivationType act = ActivationType.Linear) =>
        new(sizes, act, stateLR: 0.1f, weightLR: 0.01f, rng: new Random(42));

    // ── Constructor ───────────────────────────────────────────────────────────

    [Fact]
    public void Constructor_SingleLayer_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            new Network([4], ActivationType.Linear, 0.1f, 0.01f, new Random()));
    }

    [Fact]
    public void Constructor_TwoLayers_BuildsCorrectly()
    {
        var net = MakeNet([4, 2]);

        Assert.Equal(2, net.LayerCount);
        Assert.Equal(4, net.Layers[0].Size);
        Assert.Equal(2, net.Layers[1].Size);
    }

    [Fact]
    public void Constructor_ThreeLayers_MiddleLayerHasCorrectLowerSize()
    {
        var net = MakeNet([8, 4, 2]);

        Assert.Equal(8, net.Layers[1].LowerSize); // layer 1 predicts for layer 0
        Assert.Equal(4, net.Layers[2].LowerSize); // layer 2 predicts for layer 1
    }

    // ── Infer ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Infer_ClampsInputLayer()
    {
        var net = MakeNet([3, 2]);
        var input = new DenseTensor<float>(new float[] { 1f, 2f, 3f }, 3);

        net.Infer(input, inferenceSteps: 5);

        Assert.Equal(1f, net.Layers[0].Mu[0]);
        Assert.Equal(2f, net.Layers[0].Mu[1]);
        Assert.Equal(3f, net.Layers[0].Mu[2]);
        Assert.True(net.Layers[0].IsClamped);
    }

    [Fact]
    public void Infer_WithTopClamp_ClampsTopLayer()
    {
        var net = MakeNet([3, 2]);
        var input = new DenseTensor<float>(new float[] { 0f, 0f, 0f }, 3);
        var label = new DenseTensor<float>(new float[] { 1f, 0f }, 2);

        net.Infer(input, inferenceSteps: 3, clampTop: true, label: label);

        Assert.Equal(1f, net.Layers[^1].Mu[0]);
        Assert.Equal(0f, net.Layers[^1].Mu[1]);
        Assert.True(net.Layers[^1].IsClamped);
    }

    // ── ComputeEnergy ─────────────────────────────────────────────────────────

    [Fact]
    public void ComputeEnergy_AfterLearn_EnergyNonNegative()
    {
        var net = MakeNet([4, 3, 2]);
        var input = new DenseTensor<float>(new float[] { 1f, 0f, 0f, 0f }, 4);
        var label = new DenseTensor<float>(new float[] { 1f, 0f }, 2);

        net.Learn(input, label, inferenceSteps: 10);

        Assert.True(net.ComputeEnergy() >= 0f);
    }

    [Fact]
    public void ComputeEnergy_AfterMultipleLearns_DecreasesOnAverage()
    {
        var net = MakeNet([4, 3, 2], ActivationType.Tanh);
        var input = new DenseTensor<float>(new float[] { 1f, 0f, 0f, 0f }, 4);
        var label = new DenseTensor<float>(new float[] { 1f, 0f }, 2);

        float firstEnergy = float.MaxValue;
        for (int i = 0; i < 50; i++)
        {
            net.Learn(input, label, inferenceSteps: 5);
            if (i == 0) firstEnergy = net.ComputeEnergy();
        }

        Assert.True(net.ComputeEnergy() < firstEnergy);
    }

    // ── Predict ───────────────────────────────────────────────────────────────

    [Fact]
    public void Predict_ReturnsValidClassIndex()
    {
        var net = MakeNet([4, 3, 2]);
        var input = new DenseTensor<float>(new float[] { 1f, 0f, 0f, 0f }, 4);

        int predicted = net.Predict(input, inferenceSteps: 5);

        Assert.True(predicted >= 0 && predicted < 2);
    }

    [Fact]
    public void Predict_AfterTraining_CorrectClassForSimplePattern()
    {
        // Train a tiny net to associate input [1,0] with class 0 and [0,1] with class 1
        var net = new Network([2, 4, 2], ActivationType.Tanh, stateLR: 0.1f, weightLR: 0.05f, rng: new Random(0));

        var x0 = new DenseTensor<float>(new float[] { 1f, 0f }, 2);
        var y0 = new DenseTensor<float>(new float[] { 1f, 0f }, 2);
        var x1 = new DenseTensor<float>(new float[] { 0f, 1f }, 2);
        var y1 = new DenseTensor<float>(new float[] { 0f, 1f }, 2);

        for (int epoch = 0; epoch < 200; epoch++)
        {
            net.Learn(x0, y0, inferenceSteps: 10);
            net.Learn(x1, y1, inferenceSteps: 10);
        }

        Assert.Equal(0, net.Predict(x0, inferenceSteps: 20));
        Assert.Equal(1, net.Predict(x1, inferenceSteps: 20));
    }

    // ── AverageWeightNorm ─────────────────────────────────────────────────────

    [Fact]
    public void AverageWeightNorm_AfterInit_NonNegative()
    {
        var net = MakeNet([4, 3, 2]);
        Assert.True(net.AverageWeightNorm() >= 0f);
    }
}
