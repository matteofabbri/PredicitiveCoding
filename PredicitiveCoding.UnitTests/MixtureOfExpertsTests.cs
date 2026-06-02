using PredictiveCoding.Core;
using PredictiveCoding.Model;

namespace PredicitiveCoding.UnitTests;

public sealed class MixtureOfExpertsTests : IDisposable
{
    private readonly string _tempDir;

    public MixtureOfExpertsTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"moe_{Guid.NewGuid():N}");
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }

    private MixtureOfExperts Build(
        string sub          = "default",
        int    experts      = 2,
        float  temperature  = 1f,
        float  minGate      = 0f)
    {
        var cfg = new MixtureOfExpertsConfig
        {
            BasePath          = Path.Combine(_tempDir, sub),
            ExpertCount       = experts,
            ExpertConfig      = new NetworkConfig
            {
                LayerSizes         = [4, 8, 2],
                Activation         = ActivationType.Tanh,
                StateLearningRate  = 0.1f,
                WeightLearningRate = 0.01f,
            },
            GatingTemperature = temperature,
            MinGateWeight     = minGate,
        };
        return new MixtureOfExperts(cfg, new Random(42));
    }

    private static DenseTensor<float> OneHot(int cls, int size = 2)
    {
        var t = DenseTensor<float>.Zeros(size);
        t.Buffer[cls] = 1f;
        return t;
    }

    private static DenseTensor<float> Input(params float[] v) => new(v, v.Length);

    // ── Config validation ─────────────────────────────────────────────────────

    [Fact]
    public void Config_SingleExpert_Throws()
    {
        var cfg = new MixtureOfExpertsConfig
        {
            BasePath    = Path.Combine(_tempDir, "bad"),
            ExpertCount = 1,
            ExpertConfig = new NetworkConfig { LayerSizes = [4, 2] },
        };
        Assert.Throws<InvalidOperationException>(() => new MixtureOfExperts(cfg, new Random(0)));
    }

    [Fact]
    public void Config_ZeroTemperature_Throws()
    {
        var cfg = new MixtureOfExpertsConfig
        {
            BasePath          = Path.Combine(_tempDir, "bad_t"),
            ExpertCount       = 2,
            ExpertConfig      = new NetworkConfig { LayerSizes = [4, 2] },
            GatingTemperature = 0f,
        };
        Assert.Throws<InvalidOperationException>(() => new MixtureOfExperts(cfg, new Random(0)));
    }

    // ── Construction ─────────────────────────────────────────────────────────

    [Fact]
    public void Construction_SetsExpertCount()
    {
        using var moe = Build(experts: 3);
        Assert.Equal(3, moe.ExpertCount);
    }

    [Fact]
    public void Construction_InitialGateWeightsAreZero()
    {
        using var moe = Build();
        Assert.All(moe.LastGateWeights, g => Assert.Equal(0f, g));
    }

    // ── ComputeGates ─────────────────────────────────────────────────────────

    [Fact]
    public void ComputeGates_SumsToOne()
    {
        var energies = new float[] { 1f, 2f, 3f };
        var gates    = new float[3];
        MixtureOfExperts.ComputeGates(energies, gates, temperature: 1f, minGate: 0f);

        float sum = gates.Sum();
        Assert.Equal(1f, sum, precision: 5);
    }

    [Fact]
    public void ComputeGates_LowerEnergyGetHigherWeight()
    {
        var energies = new float[] { 0.5f, 5f };
        var gates    = new float[2];
        MixtureOfExperts.ComputeGates(energies, gates, temperature: 1f, minGate: 0f);

        Assert.True(gates[0] > gates[1],
            $"Expert 0 (lower energy) should have higher gate: {gates[0]:F4} vs {gates[1]:F4}");
    }

    [Fact]
    public void ComputeGates_HighTemperature_HarderAssignment()
    {
        var energies = new float[] { 1f, 2f };
        var gatesSoft = new float[2];
        var gatesHard = new float[2];
        MixtureOfExperts.ComputeGates(energies, gatesSoft, temperature: 0.1f, minGate: 0f);
        MixtureOfExperts.ComputeGates(energies, gatesHard, temperature: 10f,  minGate: 0f);

        // Higher temperature → more concentrated (winner closer to 1)
        Assert.True(gatesHard[0] > gatesSoft[0]);
    }

    [Fact]
    public void ComputeGates_MinGateFloor_LoserHasNonTrivialWeight()
    {
        // With extreme imbalance the raw softmax gives the loser ~0.
        // Applying the floor should give it a meaningful non-zero share.
        // Note: after one-pass floor + renorm the loser ends up at
        //   minGate / (1 + minGate) ≈ 0.09, not exactly minGate.
        float minGate = 0.1f;
        var energies  = new float[] { 0f, 100f };
        var gates     = new float[2];
        MixtureOfExperts.ComputeGates(energies, gates, temperature: 1f, minGate: minGate);

        Assert.True(gates[1] > 0.05f, $"Loser gate {gates[1]:F4} should be > 0.05 (no collapse)");
        Assert.Equal(1f, gates.Sum(), precision: 5);
    }

    [Fact]
    public void ComputeGates_EqualEnergies_UniformDistribution()
    {
        var energies = new float[] { 1f, 1f, 1f };
        var gates    = new float[3];
        MixtureOfExperts.ComputeGates(energies, gates, temperature: 1f, minGate: 0f);

        Assert.All(gates, g => Assert.Equal(1f / 3f, g, precision: 5));
    }

    // ── Learn ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Learn_PopulatesGateWeightsAndEnergies()
    {
        using var moe = Build();
        moe.Learn(Input(1f, 0f, 0f, 1f), OneHot(0), inferenceSteps: 5);

        Assert.Equal(1f, moe.LastGateWeights.Sum(),    precision: 5);
        Assert.All(moe.LastGateWeights,    g => Assert.True(g >= 0f));
        Assert.All(moe.LastExpertEnergies, e => Assert.True(e >= 0f && float.IsFinite(e)));
    }

    [Fact]
    public void Learn_ChangesWeightNorm()
    {
        using var moe = Build();
        float before = moe.AverageWeightNorm();
        moe.Learn(Input(1f, 0f, 0f, 1f), OneHot(0), inferenceSteps: 5);
        Assert.NotEqual(before, moe.AverageWeightNorm());
    }

    [Fact]
    public void Learn_RepeatedSteps_EnergyFinite()
    {
        using var moe = Build("repeat");
        for (int i = 0; i < 20; i++)
        {
            int cls = i % 2;
            moe.Learn(OneHot(cls, 4), OneHot(cls), inferenceSteps: 5);
        }
        Assert.All(moe.LastExpertEnergies, e => Assert.True(float.IsFinite(e)));
    }

    // ── Predict ───────────────────────────────────────────────────────────────

    [Fact]
    public void Predict_ReturnsValidClass()
    {
        using var moe = Build();
        int pred = moe.Predict(Input(1f, 0f, 0f, 1f), inferenceSteps: 5);
        Assert.InRange(pred, 0, 1);
    }

    [Fact]
    public void PredictDetailed_WinnerMatchesPredict()
    {
        using var moe = Build("detail");
        var input = Input(0.5f, 0.1f, 0.9f, 0.3f);
        int simple   = moe.Predict(input, inferenceSteps: 10);
        var detailed = moe.PredictDetailed(input, inferenceSteps: 10);
        Assert.Equal(simple, detailed.PredictedClass);
        Assert.InRange(detailed.WinningExpert, 0, 1);
    }

    // ── LearnGated on Network directly ───────────────────────────────────────

    [Fact]
    public void Network_LearnGated_ZeroWeight_DoesNotChangeWeights()
    {
        var cfg = new NetworkConfig
        {
            Path               = Path.Combine(_tempDir, "gated"),
            LayerSizes         = [4, 8, 2],
            Activation         = ActivationType.Tanh,
            StateLearningRate  = 0.1f,
            WeightLearningRate = 0.01f,
        };
        using var net = new Network(cfg, new Random(0));
        float before = net.AverageWeightNorm();

        net.LearnGated(Input(1f, 0f, 0f, 1f), OneHot(0), inferenceSteps: 5, gateWeight: 0f);

        Assert.Equal(before, net.AverageWeightNorm(), precision: 6);
    }

    [Fact]
    public void Network_LearnGated_ReturnsFiniteEnergy()
    {
        var cfg = new NetworkConfig
        {
            Path               = Path.Combine(_tempDir, "gated2"),
            LayerSizes         = [4, 8, 2],
            Activation         = ActivationType.Tanh,
            StateLearningRate  = 0.1f,
            WeightLearningRate = 0.01f,
        };
        using var net = new Network(cfg, new Random(0));

        float energy = net.LearnGated(Input(1f, 0f, 0f, 1f), OneHot(0), inferenceSteps: 5, gateWeight: 0.5f);

        Assert.True(energy >= 0f && float.IsFinite(energy));
    }

    [Fact]
    public void Network_InferEnergy_IsNonNegativeAndFinite()
    {
        var cfg = new NetworkConfig
        {
            Path               = Path.Combine(_tempDir, "ie"),
            LayerSizes         = [4, 8, 2],
            Activation         = ActivationType.Tanh,
            StateLearningRate  = 0.1f,
            WeightLearningRate = 0.01f,
        };
        using var net = new Network(cfg, new Random(0));

        float energy = net.InferEnergy(Input(1f, 0f, 0f, 1f), OneHot(0), inferenceSteps: 5);

        Assert.True(energy >= 0f && float.IsFinite(energy));
    }

    // ── Expert specialisation ─────────────────────────────────────────────────

    [Fact]
    public void MoE_TwoExperts_SpecialiseOnBimodalData()
    {
        // Mode A: signal in dim 0-1  (input [1,0,0,0])
        // Mode B: signal in dim 2-3  (input [0,0,1,0])
        // Both modes → same 2 labels.
        // After training, each expert should have lower gate for its off-mode.
        using var moe = Build("spec", experts: 2, temperature: 5f);
        var rng = new Random(99);

        static DenseTensor<float> ModeA(int cls)
        {
            var t = DenseTensor<float>.Zeros(4);
            t.Buffer[cls] = 1f; return t;    // dims 0-1
        }
        static DenseTensor<float> ModeB(int cls)
        {
            var t = DenseTensor<float>.Zeros(4);
            t.Buffer[cls + 2] = 1f; return t; // dims 2-3
        }

        // Train: interleave modes
        for (int step = 0; step < 800; step++)
        {
            int cls = rng.Next(2);
            bool modeA = (step % 2 == 0);
            moe.Learn(modeA ? ModeA(cls) : ModeB(cls), OneHot(cls), inferenceSteps: 10);
        }

        // Measure average gate weight for each mode
        float[] gateA = new float[2], gateB = new float[2];
        for (int t = 0; t < 50; t++)
        {
            int cls = rng.Next(2);
            moe.Learn(ModeA(cls), OneHot(cls), inferenceSteps: 10);
            for (int i = 0; i < 2; i++) gateA[i] += moe.LastGateWeights[i];

            moe.Learn(ModeB(cls), OneHot(cls), inferenceSteps: 10);
            for (int i = 0; i < 2; i++) gateB[i] += moe.LastGateWeights[i];
        }

        // The two modes should produce different gate distributions
        // (at least some asymmetry: the gate favoured by mode A ≠ that favoured by mode B)
        float asymmetry = MathF.Abs(gateA[0] / 50f - gateB[0] / 50f);
        Assert.True(asymmetry > 0.03f,
            $"Expected gate asymmetry > 0.03 between modes, got {asymmetry:F4}");
    }
}
