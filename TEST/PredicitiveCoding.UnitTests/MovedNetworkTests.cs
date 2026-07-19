using PredictiveCoding.Core;
using PredictiveCoding.Networks.MoE;
using PredictiveCoding.Networks.Ensemble;
using PredictiveCoding.Networks.Attentive;
using PredictiveCoding.Networks.Residual;
using PredictiveCoding.Networks.Routed;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

// Coverage for the advanced networks living under PredictiveCoding.Networks
// (Ensemble / Residual / Attentive / MoE.TemporalMixtureOfExperts) and Routed.
public class MovedNetworkTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"moved_{Guid.NewGuid():N}");

    public void Dispose() { if (Directory.Exists(_tmp)) Directory.Delete(_tmp, recursive: true); }

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    [Fact]
    public void EnsembleNetwork_LearnsAndPredicts()
    {
        using var ens = new EnsembleNetwork(new EnsembleConfig
        {
            MemberConfigs = [new NetworkConfig { LayerSizes = [4, 8, 2] }, new NetworkConfig { LayerSizes = [4, 6, 2] }],
            BasePath = _tmp,
        }, new Random(1));

        Assert.Equal(2, ens.MemberCount);
        var x = Vec(0.5f, -0.2f, 0.1f, 0.3f);
        ens.Learn(x, Vec(1, 0), 8);
        Assert.InRange(ens.Predict(x, 8), 0, 1);
    }

    [Fact]
    public void ResidualNetwork_LearnsAndPredicts()
    {
        using var res = new ResidualNetwork(new ResidualNetworkConfig
        {
            BaseConfig = new NetworkConfig { LayerSizes = [4, 8, 8, 2], Path = _tmp },
        }, new Random(2));

        Assert.Equal(4, res.LayerCount);
        var x = Vec(0.5f, -0.2f, 0.1f, 0.3f);
        res.Learn(x, Vec(0, 1), 8);
        Assert.InRange(res.Predict(x, 8), 0, 1);
        Assert.True(res.ComputeEnergy() >= 0f);
    }

    [Fact]
    public void AttentiveNetwork_LearnsAndPredicts()
    {
        using var att = new AttentiveNetwork(new AttentiveNetworkConfig
        {
            BaseConfig = new NetworkConfig { LayerSizes = [4, 8, 2], Path = _tmp },
        }, new Random(3));

        var x = Vec(0.5f, -0.2f, 0.1f, 0.3f);
        att.Learn(x, Vec(1, 0), 8);
        Assert.InRange(att.Predict(x, 8), 0, 1);
    }

    [Fact]
    public void TemporalMixtureOfExperts_StepsPredictsResets()
    {
        using var tmoe = new TemporalMixtureOfExperts(new TemporalMoEConfig
        {
            ExpertConfig = new NetworkConfig { LayerSizes = [4, 8, 2] }, ExpertCount = 2, BasePath = _tmp,
        }, new Random(4));

        Assert.Equal(2, tmoe.ExpertCount);
        var x = Vec(0.5f, -0.2f, 0.1f, 0.3f);
        tmoe.Step(x, Vec(1, 0), 8);
        Assert.InRange(tmoe.Predict(x, 8), 0, 1);
        tmoe.Reset();
    }

    [Fact]
    public void RoutedNetwork_LearnsAndPredicts()
    {
        using var routed = new RoutedNetwork(new RoutedNetworkConfig
        {
            ExpertConfig = new NetworkConfig { LayerSizes = [4, 8, 2] },
            RouterConfig = new NetworkConfig { LayerSizes = [4, 6, 2] },   // input 4 = expert input; output 2 = ExpertCount
            ExpertCount = 2, TopK = 2, BasePath = _tmp,
        }, new Random(5));

        Assert.Equal(2, routed.ExpertCount);
        var x = Vec(0.5f, -0.2f, 0.1f, 0.3f);
        routed.Learn(x, Vec(1, 0), 8);
        Assert.InRange(routed.Predict(x, 8), 0, 1);
    }
}
