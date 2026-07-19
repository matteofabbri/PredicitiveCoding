using PredictiveCoding.Core;
using PredictiveCoding.Model;
using PredictiveCoding.Networks;

namespace PredicitiveCoding.UnitTests;

// Mixed-precision roadmap Phase 7: sub-16-bit storage — bfloat16 (fp32 range,
// 8-bit mantissa) and int8 affine per-row (per-channel scale + zero-point).
public class MixedSub16Tests : IDisposable
{
    private readonly string _tempRoot = Path.Combine(Path.GetTempPath(), $"pcsub16_{Guid.NewGuid():N}");

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

    // ── bf16 conversions ─────────────────────────────────────────────────────

    [Fact]
    public void Bf16Conversions_RoundTripWithinTolerance_AndKeepNaN()
    {
        float[] values = [0f, 1f, -1f, 0.1f, -3.7f, 1e-3f, 12345.678f, -6.5e4f, 3e30f, float.NaN];
        var bits = new ushort[values.Length];
        var back = new float[values.Length];

        TensorUtils.FloatToBf16Bits(values, bits);
        TensorUtils.Bf16BitsToFloat(bits, back);

        for (int i = 0; i < values.Length - 1; i++)     // all finite entries
            Assert.True(MathF.Abs(back[i] - values[i]) <= MathF.Abs(values[i]) / 128f,
                $"bf16 round-trip too lossy at {values[i]}: {back[i]}");
        Assert.True(float.IsNaN(back[^1]));

        // Round-trip idempotence: a bf16-rounded value re-encodes to itself.
        var bits2 = new ushort[values.Length];
        TensorUtils.FloatToBf16Bits(back, bits2);
        for (int i = 0; i < bits.Length; i++) Assert.Equal(bits[i], bits2[i]);
    }

    // ── Serializer: bf16 records ─────────────────────────────────────────────

    [Fact]
    public void Serializer_Bf16_TwoBytesPerWeight_MastersRounded_NotMixedReady()
    {
        var layer = MakeLayer(size: 4, lowerSize: 3);

        var full = LayerSerializer.Serialize(layer);
        var bf16 = LayerSerializer.Serialize(layer, WeightStoragePrecision.BFloat16);
        Assert.Equal(3 * 4 * 2, full.Length - bf16.Length);        // same footprint as Half

        var back = LayerSerializer.AllocateAndDeserialize(bf16, 1, ActivationType.Linear);
        Assert.False(back.UsesMixedPrecision);                     // no Half copy: bf16 is a storage format

        for (int i = 0; i < layer.GenerativeWeights!.Buffer.Length; i++)
        {
            float w = layer.GenerativeWeights.Buffer[i];
            Assert.True(MathF.Abs(back.GenerativeWeights!.Buffer[i] - w) <= MathF.Abs(w) / 128f + 1e-12f,
                $"bf16 weight {i} drifted: {w} → {back.GenerativeWeights!.Buffer[i]}");
        }

        // Persisting the round-tripped layer again is bit-stable.
        var again = LayerSerializer.AllocateAndDeserialize(
            LayerSerializer.Serialize(back, WeightStoragePrecision.BFloat16), 1, ActivationType.Linear);
        for (int i = 0; i < back.GenerativeWeights!.Buffer.Length; i++)
            Assert.Equal(back.GenerativeWeights.Buffer[i], again.GenerativeWeights!.Buffer[i]);
    }

    // ── Serializer: int8 records ─────────────────────────────────────────────

    [Fact]
    public void Serializer_Int8_OneBytePerWeightPlusRowMeta_ErrorBounded()
    {
        const int Size = 6, Lower = 4;
        var layer = MakeLayer(size: Size, lowerSize: Lower);

        var full = LayerSerializer.Serialize(layer);
        var int8 = LayerSerializer.Serialize(layer, WeightStoragePrecision.Int8);
        int weightsFull = Lower * Size * 4;
        int weightsInt8 = Lower * 4 + Lower + Lower * Size;        // scales + zero-points + codes
        Assert.Equal(weightsFull - weightsInt8, full.Length - int8.Length);

        var back = LayerSerializer.AllocateAndDeserialize(int8, 1, ActivationType.Linear);
        Assert.False(back.UsesMixedPrecision);

        // Per-row error bound: |w̃ − w| ≤ s/2 with s = (max − min)/255.
        for (int r = 0; r < Lower; r++)
        {
            float min = float.MaxValue, max = float.MinValue;
            for (int j = 0; j < Size; j++)
            {
                float v = layer.GenerativeWeights!.Buffer[r * Size + j];
                if (v < min) min = v;
                if (v > max) max = v;
            }
            float bound = (max - min) / 255f * 0.5f + 1e-6f;
            for (int j = 0; j < Size; j++)
            {
                float w  = layer.GenerativeWeights!.Buffer[r * Size + j];
                float wq = back.GenerativeWeights!.Buffer[r * Size + j];
                Assert.True(MathF.Abs(wq - w) <= bound,
                    $"int8 weight [{r},{j}] out of bound: {w} → {wq} (bound {bound})");
            }
        }
    }

    // ── Network round-trip: persist sub-16, reopen, same answers ─────────────

    [Theory]
    [InlineData(WeightStoragePrecision.BFloat16)]
    [InlineData(WeightStoragePrecision.Int8)]
    public void PersistQuantized_SurvivesReopen_WithTheSameClasses(WeightStoragePrecision precision)
    {
        var cfg = new NetworkConfig
        {
            Path               = Path.Combine(_tempRoot, precision.ToString()),
            LayerSizes         = [4, 8, 2],
            Activation         = ActivationType.Tanh,
            StateLearningRate  = 0.1f,
            WeightLearningRate = 0.01f,
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
            Assert.NotEqual(clsA, clsB);

            net.PersistWeightsQuantized(precision);
            Assert.Equal(clsA, net.Predict(a, 20));                // masters already round-tripped in RAM
        }

        using var reopened = new Network(cfg, new Random(99));
        Assert.Equal(clsA, reopened.Predict(a, 20));
        Assert.Equal(clsB, reopened.Predict(b, 20));
        Assert.Equal(clsA, reopened.PredictMixed(a, 20));          // mixed path quantizes on demand
        Assert.Equal(clsB, reopened.PredictMixed(b, 20));
    }
}
