using PredictiveCoding.Core;
using PredictiveCoding.Expanded.Spiking;

namespace PredicitiveCoding.UnitTests;

public class SpikingNetworkTests
{
    private static SpikingNetwork Make() =>
        new(new SpikingNetworkConfig { LayerSizes = [4, 8, 2], Ticks = 16 }, new Random(42));

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    [Fact]
    public void Infer_ReturnsOutputLayerRates()
    {
        var net = Make();
        var r   = net.Infer(Vec(0.5f, -0.2f, 0.1f, 0.3f));
        Assert.Equal(2, r.Length);
        Assert.All(r.Buffer, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void InferEnergy_IsFiniteAndNonNegative()
    {
        var net = Make();
        float f = net.InferEnergy(Vec(0.5f, -0.2f, 0.1f, 0.3f));
        Assert.True(float.IsFinite(f));
        Assert.True(f >= 0f);
    }

    [Fact]
    public void Predict_ReturnsClassInRange_AndIsDeterministic()
    {
        var net = Make();
        var x   = Vec(0.5f, -0.2f, 0.1f, 0.3f);
        int a   = net.Predict(x);
        int b   = net.Predict(x);
        Assert.InRange(a, 0, 1);
        Assert.Equal(a, b);
    }

    [Fact]
    public void Learn_RunsWithoutError_AndKeepsEnergyFinite()
    {
        var net = Make();
        var x   = Vec(1f, 0f, -1f, 0.5f);
        var y   = Vec(0f, 1f);
        for (int i = 0; i < 20; i++) net.Learn(x, y);
        Assert.True(float.IsFinite(net.InferEnergy(x)));
    }

    [Fact]
    public void Learn_WrongLabelSize_Throws()
    {
        var net = Make();
        Assert.Throws<ArgumentException>(() => net.Learn(Vec(1, 0, -1, 0.5f), Vec(1, 0, 0)));
    }
}
