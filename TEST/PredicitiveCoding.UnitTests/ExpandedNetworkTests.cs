using PredictiveCoding.Core;
using PredictiveCoding.Networks.Generative;
using PredictiveCoding.Networks.Graph;
using PredictiveCoding.Networks.MoE;
using PredictiveCoding.Networks.Hopfield;
using PredictiveCoding.Networks.Memory;
using PredictiveCoding.Networks.MultiModal;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

public class ExpandedNetworkTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"expanded_{Guid.NewGuid():N}");

    public void Dispose() { if (Directory.Exists(_tmp)) Directory.Delete(_tmp, recursive: true); }

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    [Fact]
    public void GenerativeNetwork_EncodesDecodes_AndLearningReducesError()
    {
        using var gen = new GenerativeNetwork(new GenerativeNetworkConfig
        {
            NetworkConfig = new NetworkConfig { LayerSizes = [8, 12, 4], Activation = ActivationType.Linear, WeightLearningRate = 0.05f },
            BasePath = _tmp,
        }, new Random(1));

        var x = Vec(0.5f, -0.3f, 0.2f, 0.1f, -0.4f, 0.3f, 0.0f, 0.2f);
        Assert.Equal(4, gen.Encode(x, 12).Length);
        Assert.Equal(8, gen.Decode(Vec(0.1f, 0.2f, 0.3f, 0.4f), 12).Length);
        Assert.Equal(8, gen.Sample(12, new Random(2)).Length);

        float before = gen.ReconstructionError(x, 12);
        for (int i = 0; i < 60; i++) gen.Learn(x, 12);
        Assert.True(gen.ReconstructionError(x, 12) < before);
    }

    [Fact]
    public void HopfieldNetwork_StoresAndRecalls()
    {
        using var net = new HopfieldNetwork(new HopfieldNetworkConfig
        {
            NetworkConfig = new NetworkConfig { LayerSizes = [8, 12, 4] }, BasePath = _tmp,
        }, new Random(3));

        var x = Vec(1, 0, 1, 0, 1, 0, 1, 0);
        net.Learn(x, 10);
        Assert.Equal(4, net.Store(x, 10).Length);   // latent stored
        Assert.Equal(1, net.MemoryCount);
        Assert.Equal(8, net.Recall(x, 10).Length);   // completed input
    }

    [Fact]
    public void GraphPredictiveNetwork_LearnsCompletesAndScores()
    {
        using var g = new GraphPredictiveNetwork(new GraphPredictiveNetworkConfig
        {
            EdgePredictorConfig = new NetworkConfig { LayerSizes = [4, 8, 4] }, BasePath = _tmp,
        }, new Random(4));

        var nodes = new[] { Vec(1, 0, 0, 0), Vec(0, 1, 0, 0), Vec(0, 0, 1, 0) };
        var nbrs  = new[] { new[] { 1 }, new[] { 0, 2 }, new[] { 1 } };

        g.Learn(nodes, nbrs, 8);
        Assert.True(g.GraphEnergy(nodes, nbrs, 8) >= 0f);

        var completed = g.Complete(nodes, nbrs, [true, false, true], 8, rounds: 2);
        Assert.Equal(3, completed.Length);
        Assert.All(completed, t => Assert.Equal(4, t.Length));
    }

    [Fact]
    public void HierarchicalMixtureOfExperts_LearnsAndPredicts()
    {
        using var hmoe = new HierarchicalMixtureOfExperts(new HierarchicalMoEConfig
        {
            ExpertConfig = new NetworkConfig { LayerSizes = [4, 8, 2] },
            GroupCount = 2, ExpertsPerGroup = 2, BasePath = _tmp,
        }, new Random(5));

        var x = Vec(0.5f, -0.2f, 0.1f, 0.3f);
        hmoe.Learn(x, Vec(1, 0), 8);
        Assert.InRange(hmoe.Predict(x, 8), 0, 1);
    }

    [Fact]
    public void MultiModalNetwork_BindsAndClassifies()
    {
        using var mm = new MultiModalNetwork(new MultiModalNetworkConfig
        {
            Modalities =
            [
                new ModalityConfig { Name = "a", NetworkConfig = new NetworkConfig { LayerSizes = [6, 4] } },
                new ModalityConfig { Name = "b", NetworkConfig = new NetworkConfig { LayerSizes = [6, 4] } },
            ],
            AssociativeConfig = new NetworkConfig { LayerSizes = [8, 3] },   // 4 + 4 latents → 3 classes
            BasePath = _tmp,
        }, new Random(6));

        var inputs = new Dictionary<int, DenseTensor<float>>
        {
            [0] = Vec(1, 0, 1, 0, 1, 0),
            [1] = Vec(0, 1, 0, 1, 0, 1),
        };
        Assert.Equal(8, mm.EncodeAll(inputs, 8).Length);
        mm.Learn(inputs, Vec(1, 0, 0), 8);
        Assert.InRange(mm.Predict(inputs, 8), 0, 2);
    }

    [Fact]
    public void MemoryNetwork_LearnsClassifies_AndGrowsMemory()
    {
        using var mem = new MemoryNetwork(new MemoryNetworkConfig
        {
            NetworkConfig = new NetworkConfig { LayerSizes = [6, 8, 3] }, MemoryCapacity = 16, RetrievalTopK = 2, BasePath = _tmp,
        }, new Random(7));

        var x = Vec(0.5f, -0.2f, 0.1f, 0.3f, -0.4f, 0.2f);
        mem.Learn(x, Vec(0, 1, 0), 8);
        Assert.Equal(1, mem.MemoryCount);
        Assert.InRange(mem.Predict(x, 8), 0, 2);
    }
}
