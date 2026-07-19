using PredictiveCoding.LLM.Model;

namespace PredicitiveCoding.UnitTests;

// Phase 4 of PredictiveCoding.LLM/ROADMAP.md: long-context cache efficiency —
// sliding window + attention sinks + Half-precision storage + byte accounting.
public class LlmKvCacheTests
{
    private static TransformerConfig Cfg(int window = 0, int sinks = 0, bool half = false) => new()
    {
        VocabSize = 260, MaxSeqLen = 32, EmbedDim = 16, NumHeads = 2, NumLayers = 2,
        FeedForwardDim = 32, Positional = PositionalKind.Alibi,   // ALiBi exercises the distance bookkeeping
        WindowSize = window, NumSinkTokens = sinks, KvCacheHalf = half,
    };

    private static DenseTensorList StepMany(GptModel model, KvCache cache, int steps)
    {
        var all = new DenseTensorList();
        for (int t = 0; t < steps; t++) all.Last = model.Step(65 + t, t, cache);
        return all;
    }

    private sealed class DenseTensorList { public PredictiveCoding.Core.DenseTensor<float> Last = null!; }

    // ── Window that never fills ≡ full attention, bit for bit ────────────────

    [Fact]
    public void Window_LargerThanTheSequence_IsBitIdenticalToFull()
    {
        var full     = new GptModel(Cfg(), new Random(11));
        var windowed = new GptModel(Cfg(window: 20, sinks: 2), new Random(11));

        var cFull = full.CreateCache();
        var cWin  = windowed.CreateCache();
        for (int t = 0; t < 10; t++)
        {
            var a = full.Step(65 + t, t, cFull);
            var b = windowed.Step(65 + t, t, cWin);
            for (int i = 0; i < a.Length; i++)
                Assert.Equal(a.Buffer[i], b.Buffer[i]);   // no eviction yet ⇒ same math, same bits
        }
    }

    // ── Eviction: sinks are forever, the window slides ───────────────────────

    [Fact]
    public void Eviction_KeepsSinksAndTheLastWindowPositions()
    {
        var model = new GptModel(Cfg(window: 4, sinks: 2), new Random(3));
        var cache = model.CreateCache();
        StepMany(model, cache, 10);                       // positions 0..9

        var layer = (KvCacheLayer)cache.Layers[0];
        Assert.Equal(6, layer.Length);                    // 2 sinks + 4 window
        Assert.Equal(0, layer.Position(0));               // sinks survive…
        Assert.Equal(1, layer.Position(1));
        Assert.Equal(6, layer.Position(2));               // …the window holds the last 4
        Assert.Equal(9, layer.Position(5));
    }

    [Fact]
    public void BytesUsed_PlateausOnceTheWindowIsFull()
    {
        var model = new GptModel(Cfg(window: 4, sinks: 1), new Random(4));
        var cache = model.CreateCache();

        StepMany(model, cache, 5);                        // exactly sinks + window
        long atCapacity = cache.BytesUsed;
        Assert.True(atCapacity > 0);

        StepMany(model, cache, 10);                       // keep decoding well past it
        Assert.Equal(atCapacity, cache.BytesUsed);        // fixed budget, as promised
    }

    // ── Half-precision storage ───────────────────────────────────────────────

    [Fact]
    public void HalfCache_HalvesTheBytes_AndStaysNumericallyClose()
    {
        var f32 = new GptModel(Cfg(), new Random(7));
        var f16 = new GptModel(Cfg(half: true), new Random(7));   // same seed ⇒ same weights

        var c32 = f32.CreateCache();
        var c16 = f16.CreateCache();
        PredictiveCoding.Core.DenseTensor<float> a = null!, b = null!;
        for (int t = 0; t < 8; t++)
        {
            a = f32.Step(65 + t, t, c32);
            b = f16.Step(65 + t, t, c16);
        }

        Assert.Equal(c32.BytesUsed, c16.BytesUsed * 2);   // exactly half the storage

        float maxAbs = 0f, maxDiff = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            maxAbs  = MathF.Max(maxAbs, MathF.Abs(a.Buffer[i]));
            maxDiff = MathF.Max(maxDiff, MathF.Abs(a.Buffer[i] - b.Buffer[i]));
        }
        Assert.True(maxDiff <= 0.05f * MathF.Max(maxAbs, 1f),
            $"Half cache drifted too far: maxDiff {maxDiff:F4} vs maxAbs {maxAbs:F2}");
    }

    // ── Accounting on the compressed (MLA) cache too ─────────────────────────

    [Fact]
    public void LatentCache_ReportsItsBytes()
    {
        var cfg = new TransformerConfig
        {
            VocabSize = 260, MaxSeqLen = 16, EmbedDim = 16, NumHeads = 2, NumLayers = 1,
            FeedForwardDim = 32, Attention = AttentionKind.Latent,
            KvLatentDim = 8, RopeHeadDim = 4, Positional = PositionalKind.Rotary,
        };
        var model = new GptModel(cfg, new Random(9));
        var cache = model.CreateCache();
        model.Step(65, 0, cache);

        Assert.Equal((8 + 4) * sizeof(float), cache.BytesUsed);   // c_kv + k_R, one layer, one position
    }

    // ── Config validation ────────────────────────────────────────────────────

    [Fact]
    public void InvalidWindowConfigs_Throw()
    {
        Assert.Throws<InvalidOperationException>(() => Cfg(window: 0, sinks: 2).Validate());
        Assert.Throws<InvalidOperationException>(() => (Cfg(window: 4) with { Attention = AttentionKind.Latent }).Validate());
    }
}
