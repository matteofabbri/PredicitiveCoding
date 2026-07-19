using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Infrastructure;
using PredictiveCoding.LLM.Training;

namespace PredicitiveCoding.UnitTests;

// ROADMAP-MixedPrecision Phase 5, "route GEMM-heavy consumers to mixed":
// Ag.MatMulRoute sends the trainable stack's MatMul forwards through the
// pluggable Gemm.Provider — full float, or the mixed path (16-bit operands,
// fp32 accumulation). Backwards always stay inline fp32 (master-precision
// gradients). These tests mutate the two process-wide knobs (route, provider),
// so the collection is serialised against the rest of the suite.
[CollectionDefinition("gemm-route", DisableParallelization = true)]
public sealed class GemmRouteCollection { }

[Collection("gemm-route")]
public class LlmGemmRouteTests
{
    private static DecoderLmConfig Cfg() =>
        new(13, 8, NumBlocks: 2, NumHeads: 2, FfDim: 16, MaxSeq: 8);

    /// <summary>Counts provider calls while delegating to the real CPU backend.</summary>
    private sealed class SpyProvider : IGemmProvider
    {
        private readonly CpuGemmProvider _inner = new();
        public int FloatCalls;
        public int MixedCalls;

        public float[] MatMul(float[] a, int m, int k, float[] b, int n)
        {
            Interlocked.Increment(ref FloatCalls);
            return _inner.MatMul(a, m, k, b, n);
        }

        public float[] MatMulMixed(float[] a, int m, int k, float[] b, int n)
        {
            Interlocked.Increment(ref MixedCalls);
            return _inner.MatMulMixed(a, m, k, b, n);
        }
    }

    private static void WithRoute(MatMulRoute route, SpyProvider spy, Action body)
    {
        var keepProvider = Gemm.Provider;
        var keepRoute    = Ag.MatMulRoute;
        Gemm.Provider    = spy;
        Ag.MatMulRoute   = route;
        try { body(); }
        finally
        {
            Gemm.Provider  = keepProvider;
            Ag.MatMulRoute = keepRoute;
        }
    }

    // ── Off means off: the default route never touches the provider ──────────

    [Fact]
    public void InlineRoute_NeverTouchesTheProvider()
    {
        var spy = new SpyProvider();
        WithRoute(MatMulRoute.Inline, spy, () =>
            new DecoderLM(Cfg(), new Random(3)).Forward([1, 2, 3, 4]));

        Assert.Equal(0, spy.FloatCalls);
        Assert.Equal(0, spy.MixedCalls);
    }

    // ── Float routing changes the engine, not the numbers ────────────────────

    [Fact]
    public void ProviderRoute_MatchesInline_BitForBit()
    {
        var lm     = new DecoderLM(Cfg(), new Random(5));
        int[] toks = [1, 2, 3, 4, 5];
        var inline = lm.Forward(toks);

        var spy = new SpyProvider();
        Tensor routed = null!;
        WithRoute(MatMulRoute.Provider, spy, () => routed = lm.Forward(toks));

        Assert.True(spy.FloatCalls > 0, "the provider was never consulted");
        Assert.Equal(0, spy.MixedCalls);
        Assert.Equal(inline.Data, routed.Data);        // same i-k-j numerics, bit for bit
    }

    // ── Mixed routing calls the mixed kernel and tracks float closely ────────

    [Fact]
    public void MixedRoute_CallsMixedKernel_AndTracksFloatWithinTolerance()
    {
        var lm     = new DecoderLM(Cfg(), new Random(7));
        int[] toks = [1, 2, 3, 4, 5];
        var full   = lm.Forward(toks);

        var spy = new SpyProvider();
        Tensor mixed = null!;
        WithRoute(MatMulRoute.ProviderMixed, spy, () => mixed = lm.Forward(toks));

        Assert.True(spy.MixedCalls > 0, "the mixed kernel was never consulted");
        Assert.Equal(0, spy.FloatCalls);
        for (int i = 0; i < full.Data.Length; i++)
        {
            float tol = 3e-2f * (1f + MathF.Abs(full.Data[i]));   // Half rounding, compounded over blocks
            Assert.True(MathF.Abs(full.Data[i] - mixed.Data[i]) <= tol,
                $"logit {i}: float={full.Data[i]}, mixed={mixed.Data[i]}");
        }
    }

    // ── Mixed forward + fp32 backward still trains ───────────────────────────

    [Fact]
    public void MixedRoute_Training_ReducesLoss()
    {
        var spy = new SpyProvider();
        WithRoute(MatMulRoute.ProviderMixed, spy, () =>
        {
            var lm  = new DecoderLM(Cfg(), new Random(4));
            var opt = new Adam();
            int[] toks = [1, 2, 3, 4, 5], tgts = [2, 3, 4, 5, 6];

            float first = lm.Loss(toks, tgts, ignoreIndex: -1).Data[0];
            for (int it = 0; it < 150; it++)
            {
                lm.Loss(toks, tgts, ignoreIndex: -1, training: true).Backward();
                opt.Step(lm.Parameters(), 0.01f);
            }
            float last = lm.Loss(toks, tgts, ignoreIndex: -1).Data[0];

            Assert.True(last < first * 0.5f, $"mixed-route model did not learn: {first:F3} → {last:F3}");
        });
        Assert.True(spy.MixedCalls > 0);
    }
}
