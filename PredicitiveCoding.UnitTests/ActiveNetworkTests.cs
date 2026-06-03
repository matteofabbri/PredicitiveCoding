using PredictiveCoding.Core;
using PredictiveCoding.Expanded.Active;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

public class ActiveNetworkTests : IDisposable
{
    private readonly string _tempDir;

    public ActiveNetworkTests()
        => _tempDir = Path.Combine(Path.GetTempPath(), $"pcactive_{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, recursive: true);
    }

    private ActiveNetwork Make() => new(
        new ActiveNetworkConfig
        {
            ObservationDim = 3,
            ActionCount    = 2,
            WorldModelConfig = new NetworkConfig { LayerSizes = [5, 8, 3], WeightLearningRate = 0.05f },
            BasePath = _tempDir,
        },
        new Random(42));

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);

    private static float Err(DenseTensor<float> a, DenseTensor<float> b)
    {
        float s = 0f;
        for (int i = 0; i < a.Length; i++) { float d = a.Buffer[i] - b.Buffer[i]; s += d * d; }
        return 0.5f * s;
    }

    [Fact]
    public void PredictNext_HasObservationShape()
    {
        using var agent = Make();
        Assert.Equal(3, agent.PredictNext(Vec(0.1f, 0.2f, -0.3f), action: 0, 8).Length);
    }

    [Fact]
    public void SelectAction_ReturnsValidActionAndScores()
    {
        using var agent = Make();
        var (action, scores) = agent.SelectAction(Vec(0.1f, 0.2f, -0.3f), Vec(0.5f, 0.5f, 0.5f), 8);
        Assert.InRange(action, 0, 1);
        Assert.Equal(2, scores.Length);
    }

    [Fact]
    public void Learning_ImprovesTransitionPrediction()
    {
        using var agent = Make();
        var obs    = Vec(0.2f, -0.1f, 0.3f);
        var target = Vec(0.5f, -0.5f, 0.2f);

        float before = Err(agent.PredictNext(obs, 0, 12), target);
        for (int i = 0; i < 120; i++) agent.Observe(obs, 0, target, 12);
        float after = Err(agent.PredictNext(obs, 0, 12), target);

        Assert.True(after < before, $"world model should improve (before={before}, after={after}).");
    }

    [Fact]
    public void Observe_WrongNextStateSize_Throws()
    {
        using var agent = Make();
        Assert.Throws<ArgumentException>(() => agent.Observe(Vec(0.2f, -0.1f, 0.3f), 0, Vec(1, 2), 8));
    }
}
