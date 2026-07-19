using PredictiveCoding.Core;
using PredictiveCoding.Model;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

// Mixed-precision roadmap Phase 6: model I/O — weights persisted as Half
// (half the record size), reloaded mixed-ready (no re-quantizing).
public class MixedPersistenceTests : IDisposable
{
    private readonly string _tempRoot = Path.Combine(Path.GetTempPath(), $"pcmixio_{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_tempRoot)) Directory.Delete(_tempRoot, recursive: true);
    }

    private static Layer MakeLayer(int size, int lowerSize)
    {
        var layer = new Layer
        {
            Index              = 1,
            Size               = size,
            LowerSize          = lowerSize,
            ActivationType     = ActivationType.Linear,
            StateLearningRate  = 0.1f,
            WeightLearningRate = 0.01f,
        };
        layer.Initialize(new Random(42));
        return layer;
    }

    // ── Serializer: the record shrinks and comes back mixed-ready ────────────

    [Fact]
    public void Serializer_HalfFlag_ShrinksTheRecord_AndRoundTripsMixedReady()
    {
        var layer = MakeLayer(size: 4, lowerSize: 3);

        var full = LayerSerializer.Serialize(layer);
        var half = LayerSerializer.Serialize(layer, halfWeights: true);
        Assert.Equal(3 * 4 * 2, full.Length - half.Length);        // 12 weights × 2 bytes saved each

        var back = LayerSerializer.AllocateAndDeserialize(half, index: 1, ActivationType.Linear);

        Assert.True(back.UsesMixedPrecision);                      // the Half copy ships with the record
        for (int i = 0; i < layer.GenerativeWeights!.Buffer.Length; i++)
        {
            float rounded = (float)(Half)layer.GenerativeWeights.Buffer[i];
            Assert.Equal(rounded, back.GenerativeWeights!.Buffer[i]);              // masters = widened Half
            Assert.Equal(rounded, (float)back.GenerativeWeightsHalf!.Buffer[i]);   // copy = exactly the stored bits
        }
        for (int i = 0; i < layer.Bias!.Length; i++)
            Assert.Equal(layer.Bias.Buffer[i], back.Bias!.Buffer[i]);              // everything else stays float
        for (int i = 0; i < layer.Mu.Length; i++)
            Assert.Equal(layer.Mu.Buffer[i], back.Mu.Buffer[i]);
    }

    [Fact]
    public void Serializer_FloatPath_IsUntouchedByDefault()
    {
        var layer = MakeLayer(size: 4, lowerSize: 3);
        var back  = LayerSerializer.AllocateAndDeserialize(LayerSerializer.Serialize(layer), 1, ActivationType.Linear);

        Assert.False(back.UsesMixedPrecision);
        for (int i = 0; i < layer.GenerativeWeights!.Buffer.Length; i++)
            Assert.Equal(layer.GenerativeWeights.Buffer[i], back.GenerativeWeights!.Buffer[i]);   // bit-exact
    }

    // ── QuantizeWeightsToHalf: skip when ready, force to re-sync ─────────────

    [Fact]
    public void QuantizeWeightsToHalf_SkipsWhenPresent_ForceResyncs()
    {
        var layer = MakeLayer(size: 4, lowerSize: 3);
        layer.QuantizeWeightsToHalf();
        float original = (float)layer.GenerativeWeightsHalf!.Buffer[0];

        layer.GenerativeWeights!.Buffer[0] = 123.456f;
        layer.QuantizeWeightsToHalf();                             // copy present ⇒ no re-quantize
        Assert.Equal(original, (float)layer.GenerativeWeightsHalf!.Buffer[0]);

        layer.QuantizeWeightsToHalf(force: true);                  // explicit re-sync
        Assert.Equal((float)(Half)123.456f, (float)layer.GenerativeWeightsHalf!.Buffer[0]);
    }

    // ── Network: persist as Half, reopen, same behaviour, mixed-ready ────────

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void PersistWeightsAsHalf_SurvivesReopen_WithTheSameBehaviour(bool lazy)
    {
        var cfg = new NetworkConfig
        {
            Path               = Path.Combine(_tempRoot, lazy ? "lazy" : "eager"),
            LayerSizes         = [4, 8, 2],
            Activation         = ActivationType.Tanh,
            StateLearningRate  = 0.1f,
            WeightLearningRate = 0.01f,
            LazyPersistence    = lazy,
        };
        var a = new DenseTensor<float>([1f, 0f, 0f, 0f], 4);
        var b = new DenseTensor<float>([0f, 0f, 0f, 1f], 4);
        var oneHotA = DenseTensor<float>.Zeros(2); oneHotA.Buffer[0] = 1f;
        var oneHotB = DenseTensor<float>.Zeros(2); oneHotB.Buffer[1] = 1f;

        int clsA, clsB;
        using (var net = new Network(cfg, new Random(42)))
        {
            for (int epoch = 0; epoch < 40; epoch++)
            {
                net.Learn(a, oneHotA, 30);
                net.Learn(b, oneHotB, 30);
            }
            clsA = net.Predict(a, 20);
            clsB = net.Predict(b, 20);
            Assert.NotEqual(clsA, clsB);                           // the net actually learned two classes

            net.PersistWeightsAsHalf();                            // last write: the store now holds Half records
            Assert.Equal(clsA, net.PredictMixed(a, 20));           // masters rounded in RAM too — same answer
        }

        // Reopen from disk with a DIFFERENT seed: everything must come from the
        // Half-persisted records, already mixed-ready.
        using var reopened = new Network(cfg, new Random(99));
        Assert.Equal(clsA, reopened.PredictMixed(a, 20));
        Assert.Equal(clsB, reopened.PredictMixed(b, 20));
        Assert.Equal(clsA, reopened.Predict(a, 20));               // float path = widened Half masters
        Assert.Equal(clsB, reopened.Predict(b, 20));
    }
}
