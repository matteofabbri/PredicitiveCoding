using PredictiveCoding.Core;
using PredictiveCoding.Cortex;
using PredictiveCoding.Networks.Active;
using PredictiveCoding.Networks.DeepRecurrent;
using PredictiveCoding.Networks.Gated;
using PredictiveCoding.Networks.Generative;
using PredictiveCoding.Networks.Hierarchical;
using PredictiveCoding.Networks.MoE;
using PredictiveCoding.Networks.MultiModal;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

// Coverage for the cortex modules that wrap already-tested structures
// (DefaultMode, Motor, Association, Somatosensory, Prefrontal, Dorsal stream).
public class CortexWrapperTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"cortexw_{Guid.NewGuid():N}");

    public void Dispose() { if (Directory.Exists(_tmp)) Directory.Delete(_tmp, recursive: true); }

    private static DenseTensor<float> Vec(params float[] v) => new(v, v.Length);
    private static DenseTensor<float> Image(int c, int h, int w)
    {
        var t = new DenseTensor<float>(c, h, w);
        for (int i = 0; i < t.Length; i++) t.Buffer[i] = (i % 5 - 2) * 0.2f;
        return t;
    }

    [Fact]
    public void DefaultModeNetwork_LearnsAndImagines()
    {
        using var dmn = new DefaultModeNetwork(new DefaultModeNetworkConfig
        {
            Generative = new GenerativeNetworkConfig { NetworkConfig = new NetworkConfig { LayerSizes = [8, 12, 4] } },
            BasePath = _tmp,
        }, new Random(1));

        dmn.LearnFromExperience(Vec(0.1f, 0.2f, 0.3f, 0.4f, 0.1f, 0.2f, 0.3f, 0.4f), 10);
        Assert.Equal(8, dmn.Imagine(10, new Random(2)).Length);
        Assert.Equal(3, dmn.ImagineSequence(length: 3, steps: 10, new Random(3)).Count);
    }

    [Fact]
    public void MotorCortex_PredictsSelectsAndLearns()
    {
        using var motor = new MotorCortex(new MotorCortexConfig
        {
            Agent = new ActiveNetworkConfig { ObservationDim = 2, ActionCount = 2, WorldModelConfig = new NetworkConfig { LayerSizes = [4, 8, 2] } },
            BasePath = _tmp,
        }, new Random(2));

        var s = Vec(0.2f, 0.3f);
        Assert.Equal(2, motor.PredictOutcome(s, 0, 8).Length);
        Assert.InRange(motor.SelectCommand(s, Vec(0.8f, 0.8f), 8), 0, 1);
        motor.Learn(s, 0, Vec(0.3f, 0.4f), 8);
    }

    [Fact]
    public void AssociationCortex_BindsAndClassifies()
    {
        using var assoc = new AssociationCortex(new AssociationCortexConfig
        {
            MultiModal = new MultiModalNetworkConfig
            {
                Modalities = [new ModalityConfig { Name = "a", NetworkConfig = new NetworkConfig { LayerSizes = [6, 4] } },
                              new ModalityConfig { Name = "b", NetworkConfig = new NetworkConfig { LayerSizes = [6, 4] } }],
                AssociativeConfig = new NetworkConfig { LayerSizes = [8, 3] },
            },
            BasePath = _tmp,
        }, new Random(3));

        var inputs = new Dictionary<int, DenseTensor<float>> { [0] = Vec(1, 0, 1, 0, 1, 0), [1] = Vec(0, 1, 0, 1, 0, 1) };
        Assert.Equal(8, assoc.Bind(inputs, 8).Length);
        assoc.Learn(inputs, Vec(1, 0, 0), 8);
        Assert.InRange(assoc.Predict(inputs, 8), 0, 2);
    }

    [Fact]
    public void SomatosensoryCortex_EncodesAndRecognises()
    {
        using var s1 = new SomatosensoryCortex(new SomatosensoryCortexConfig
        {
            ReceptorChannels = 1, MapHeight = 4, MapWidth = 4, PatchSize = 2, PatchStride = 2,
            S1Column = new NetworkConfig { LayerSizes = [4, 8, 4] },          // 1·2·2 = 4 → K = 4
            HigherAreas = new HierarchicalNetworkConfig { Modules = [new NetworkConfig { LayerSizes = [2 * 2 * 4, 8] }, new NetworkConfig { LayerSizes = [8, 4] }] },
            Classifier = new NetworkConfig { LayerSizes = [4, 3] },
            BasePath = _tmp,
        }, new Random(4));

        var skin = Image(1, 4, 4);
        Assert.Equal(4, s1.Encode(skin, 6).Length);
        s1.LearnSupervised(skin, Vec(0, 1, 0), 6);
        Assert.InRange(s1.Recognize(skin, 6), 0, 2);
    }

    [Fact]
    public void PrefrontalCortex_MaintainsMemoryAndDecides()
    {
        using var pfc = new PrefrontalCortex(new PrefrontalCortexConfig
        {
            WorkingMemory = new GatedStateNetworkConfig { NetworkConfig = new NetworkConfig { LayerSizes = [4, 8, 3] } },
            TaskExperts   = new HierarchicalMoEConfig { ExpertConfig = new NetworkConfig { LayerSizes = [3, 8, 2] }, GroupCount = 2, ExpertsPerGroup = 2 },
            BasePath = _tmp,
        }, new Random(5));

        var x = Vec(0.5f, -0.2f, 0.1f, 0.3f);
        Assert.Equal(3, pfc.Step(x, 8).Length);
        pfc.Learn(x, Vec(1, 0), 8);
        Assert.InRange(pfc.Decide(x, 8), 0, 1);
        pfc.Reset();
    }

    [Fact]
    public void DorsalStream_LearnsSequenceAndPredictsMotion()
    {
        using var dorsal = new DorsalStream(new DorsalStreamConfig
        {
            Spatial = new VisualCortexConfig
            {
                ImageChannels = 1, ImageHeight = 4, ImageWidth = 4, PatchSize = 2, PatchStride = 2,
                V1Column = new NetworkConfig { LayerSizes = [4, 8, 4] },
                HigherAreas = new HierarchicalNetworkConfig { Modules = [new NetworkConfig { LayerSizes = [2 * 2 * 4, 8] }, new NetworkConfig { LayerSizes = [8, 4] }] },
            },
            Temporal = new DeepRecurrentNetworkConfig { Stages = [new NetworkConfig { LayerSizes = [4, 6, 3] }, new NetworkConfig { LayerSizes = [3, 5, 2] }] },
            MotionReadout = new NetworkConfig { LayerSizes = [2, 3] },   // temporal top (2) → 3 motion classes
            BasePath = _tmp,
        }, new Random(6));

        var frames = new[] { Image(1, 4, 4), Image(1, 4, 4) };
        dorsal.LearnSequence(frames, Vec(1, 0, 0), 5);
        Assert.InRange(dorsal.PredictMotion(frames, 5), 0, 2);
    }
}
