using System.Diagnostics;
using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Infrastructure;
using PredictiveCoding.LLM.Training;

namespace PredicitiveCoding.UnitTests;

// Pro roadmap Phase 3: training throughput. Parallel micro-batch with a
// deterministic gradient merge (1 worker ≡ serial bit-for-bit; N workers
// reproducible), plus the size-adaptive Auto GEMM route.
public class LlmProPhase3Tests
{
    private static DecoderLmConfig Cfg() =>
        new(260, Dim: 16, NumBlocks: 2, NumHeads: 2, FfDim: 32, MaxSeq: 48, BlockPattern: "TA");

    private static (Adam opt, DecoderLM model) Fresh(int seed = 7)
    {
        var model = new DecoderLM(Cfg(), new Random(seed));
        return (new Adam(), model);
    }

    private static List<(int[] tokens, int[] targets)> Batch()
    {
        var batch = new List<(int[], int[])>();
        var rng = new Random(123);
        for (int i = 0; i < 8; i++)
        {
            var toks = new int[12];
            var tgts = new int[12];
            for (int j = 0; j < 12; j++) { toks[j] = rng.Next(0, 256); tgts[j] = rng.Next(0, 256); }
            batch.Add((toks, tgts));
        }
        return batch;
    }

    // ── One worker is the serial path, bit for bit ──────────────────────────

    [Fact]
    public void SingleWorker_IsBitIdenticalToSerialStep()
    {
        var batch = Batch();

        var serial = new DecoderLM(Cfg(), new Random(7));
        var sOpt   = new Adam();
        float sLoss = Trainer.Step(serial.Parameters(), batch,
            (t, y, tr) => serial.Loss(t, y, -1, tr), sOpt, 0.01f, clipNorm: 1f);

        var par    = new DecoderLM(Cfg(), new Random(7));
        var pOpt   = new Adam();
        float pLoss = Trainer.StepParallel(
            [par.Parameters()],
            [(t, y, tr) => par.Loss(t, y, -1, tr)],
            batch, pOpt, 0.01f, clipNorm: 1f);

        Assert.Equal(sLoss, pLoss);
        var sp = serial.Parameters(); var pp = par.Parameters();
        for (int k = 0; k < sp.Length; k++)
            for (int i = 0; i < sp[k].Data.Length; i++)
                Assert.Equal(sp[k].Data[i], pp[k].Data[i]);   // weights identical after the step
    }

    // ── Many workers: reproducible, and it still learns ─────────────────────

    [Fact]
    public void ParallelStep_IsReproducibleAcrossRuns()
    {
        float RunFour()
        {
            var models = new[] { new DecoderLM(Cfg(), new Random(7)),
                                 new DecoderLM(Cfg(), new Random(8)),
                                 new DecoderLM(Cfg(), new Random(9)),
                                 new DecoderLM(Cfg(), new Random(10)) };
            var opt = new Adam();
            var batch = Batch();
            float loss = 0f;
            for (int s = 0; s < 3; s++)
                loss = Trainer.StepParallel(
                    [.. models.Select(m => m.Parameters())],
                    [.. models.Select(m => (Func<int[], int[], bool, Tensor>)((t, y, tr) => m.Loss(t, y, -1, tr)))],
                    batch, opt, 0.01f, clipNorm: 1f);
            return loss;
        }

        Assert.Equal(RunFour(), RunFour());   // deterministic reduction order → identical across runs
    }

    [Fact]
    public void ParallelTraining_ReducesLossLikeSerial()
    {
        const string corpus = "the quick brown fox jumps over the lazy dog\n" +
                              "the quick brown fox jumps over the lazy dog\n" +
                              "pack my box with five dozen liquor jugs";
        var opts = new TrainingOptions { SeqLen = 16, Stride = 8, Epochs = 8, BatchWindows = 6, Workers = 4 };

        var llm    = PredictiveLlm.Create2026(dim: 16, blocks: 2, heads: 2, ffDim: 32, maxSeq: 48);
        var report = llm.Train(corpus, opts);

        Assert.True(report.LastLoss < report.FirstLoss,
            $"parallel training did not reduce loss: {report.FirstLoss:F3} → {report.LastLoss:F3}");
    }

    // ── Auto GEMM route: inline below the crossover, provider above ─────────

    [Fact]
    public void AutoRoute_MatchesInline_AndCrossesOverBySize()
    {
        var spy = new SpyProvider();
        var keepProvider = Gemm.Provider;
        var keepRoute    = Ag.MatMulRoute;
        var keepCross    = Ag.ProviderCrossoverMacs;
        Gemm.Provider  = spy;
        Ag.MatMulRoute = MatMulRoute.Auto;
        try
        {
            var rng = new Random(1);
            Tensor Rand(int r, int c)
            {
                var t = new Tensor(r, c);
                for (int i = 0; i < t.Data.Length; i++) t.Data[i] = (float)(rng.NextDouble() * 2 - 1);
                return t;
            }

            // Small product (8·8·8 = 512 MACs) stays inline: provider untouched.
            Ag.ProviderCrossoverMacs = 1L << 20;
            spy.Reset();
            var small = Ag.MatMul(Rand(8, 8), Rand(8, 8));
            Assert.Equal(0, spy.FloatCalls);

            // Force a low crossover so the same product routes to the provider,
            // and confirm the numbers match the inline path bit for bit (CPU
            // provider ≡ inline i-k-j accumulation).
            var a = Rand(8, 8); var b = Rand(8, 8);
            Ag.MatMulRoute = MatMulRoute.Inline;
            var inline = Ag.MatMul(a, b);
            Ag.MatMulRoute = MatMulRoute.Auto;
            Ag.ProviderCrossoverMacs = 1;
            spy.Reset();
            var routed = Ag.MatMul(a, b);
            Assert.True(spy.FloatCalls > 0, "Auto did not route to the provider above the crossover");
            Assert.Equal(inline.Data, routed.Data);
        }
        finally
        {
            Gemm.Provider = keepProvider;
            Ag.MatMulRoute = keepRoute;
            Ag.ProviderCrossoverMacs = keepCross;
        }
    }

    private sealed class SpyProvider : IGemmProvider
    {
        private readonly CpuGemmProvider _inner = new();
        public int FloatCalls;
        public void Reset() => FloatCalls = 0;
        public float[] MatMul(float[] a, int m, int k, float[] b, int n)
        {
            Interlocked.Increment(ref FloatCalls);
            return _inner.MatMul(a, m, k, b, n);
        }
        public float[] MatMulMixed(float[] a, int m, int k, float[] b, int n) => _inner.MatMulMixed(a, m, k, b, n);
    }

    // ── Throughput yardstick (informational; asserts only correctness) ──────

    [Fact]
    public void ParallelTraining_ThroughputYardstick()
    {
        const string corpus = "the quick brown fox jumps over the lazy dog\n" +
                              "pack my box with five dozen liquor jugs\n" +
                              "sphinx of black quartz judge my vow\n" +
                              "how vexingly quick daft zebras jump";
        var baseOpts = new TrainingOptions { SeqLen = 24, Stride = 6, Epochs = 4, BatchWindows = 8 };

        double serialMs = Time(() =>
            PredictiveLlm.Create2026(dim: 24, blocks: 3, heads: 3, ffDim: 48, maxSeq: 64)
                         .Train(corpus, baseOpts with { Workers = 1 }));
        double parMs = Time(() =>
            PredictiveLlm.Create2026(dim: 24, blocks: 3, heads: 3, ffDim: 48, maxSeq: 64)
                         .Train(corpus, baseOpts with { Workers = 0 }));

        // Not asserted as a hard speedup — CI cores vary and the toy model is
        // small — but both must produce a finite run; the ratio is logged.
        Assert.True(serialMs > 0 && parMs > 0);
        Console.WriteLine($"[Pro P3 yardstick] serial {serialMs:F0} ms, parallel {parMs:F0} ms, " +
                          $"speedup ×{serialMs / parMs:F2} on {Environment.ProcessorCount} cores");
    }

    private static double Time(Action body)
    {
        var sw = Stopwatch.StartNew();
        body();
        sw.Stop();
        return sw.Elapsed.TotalMilliseconds;
    }
}
