using System.Diagnostics;
using PredictiveCoding.Core;
using PredictiveCoding.Device;
using PredictiveCoding.LLM.Infrastructure;
using PredictiveCoding.Networks;

// ─────────────────────────────────────────────────────────────────────────────
// MixedPrecisionBench  —  ROADMAP-MixedPrecision Phase 4: put real numbers on
// the float32-vs-mixed trade, in-house (Stopwatch, warmup + median — no
// benchmarking dependency; a memory/bandwidth story does not need
// BenchmarkDotNet-grade rigor).
//
//   1. MatVecMul / MatVecMulT   float vs mixed (Half weights, fp32 accumulate)
//   2. Network inference        InferGetTop vs InferGetTopMixed (CIFAR-scale)
//   3. GEMM 1024³               CPU float vs CPU mixed vs CUDA mixed (if found)
//
// The CUDA row needs the CUDA 12 runtime DLLs discoverable (Toolkit, or e.g.
// LM Studio's bundled copies on PATH) — otherwise it reports the CPU fallback.
// ─────────────────────────────────────────────────────────────────────────────

Console.WriteLine("═══ Mixed-precision benchmark (median of 5, after warmup) ═══\n");

// ── 1. MatVec: the predictive-coding forward primitive ───────────────────────
Console.WriteLine("| op | size | float32 | mixed (Half W) | W storage |");
Console.WriteLine("|----|------|--------:|---------------:|----------:|");

foreach (int size in new[] { 512, 2048 })
{
    var rng = new Random(1);
    var w   = RandomUtils.SampleNormal([size, size], 0f, 0.05f, rng);
    var wH  = TensorUtils.ToHalf(w);
    var v   = RandomUtils.SampleNormal([size], 0f, 1f, rng);
    int k   = size <= 512 ? 2000 : 200;                       // calls per measured rep

    double f  = MedianMs(() => { for (int i = 0; i < k; i++) TensorUtils.MatVecMul(w, v); }) / k;
    double mx = MedianMs(() => { for (int i = 0; i < k; i++) TensorUtils.MatVecMulMixed(wH, v); }) / k;
    Console.WriteLine($"| MatVecMul | {size}×{size} | {Us(f)} | {Us(mx)} | {Mb(size * size * 4)} → {Mb(size * size * 2)} |");

    double ft  = MedianMs(() => { for (int i = 0; i < k; i++) TensorUtils.MatVecMulT(w, v); }) / k;
    double mxt = MedianMs(() => { for (int i = 0; i < k; i++) TensorUtils.MatVecMulTMixed(wH, v); }) / k;
    Console.WriteLine($"| MatVecMulT | {size}×{size} | {Us(ft)} | {Us(mxt)} | idem |");
}

// ── 2. Network-level inference (CIFAR-scale PC net) ──────────────────────────
string tmp = Path.Combine(Path.GetTempPath(), $"mixedbench_{Guid.NewGuid():N}");
try
{
    var cfg = new NetworkConfig
    {
        Path              = tmp,
        LayerSizes        = [3072, 512, 128, 10],
        Activation        = ActivationType.Tanh,
        StateLearningRate = 0.05f,
        LazyPersistence   = true,
    };
    using var net = new Network(cfg, new Random(0));
    var x = RandomUtils.SampleNormal([3072], 0f, 1f, new Random(2));

    long wFloat = 0;
    for (int l = 1; l < cfg.LayerSizes.Length; l++)
        wFloat += (long)cfg.LayerSizes[l - 1] * cfg.LayerSizes[l] * 4;

    double nf = MedianMs(() => net.InferGetTop(x, inferenceSteps: 20, resetState: true));
    double nm = MedianMs(() => net.InferGetTopMixed(x, inferenceSteps: 20, resetState: true));
    Console.WriteLine($"\n| Network [3072,512,128,10], 20 steps | float32 | mixed | weights |");
    Console.WriteLine($"|---|--------:|------:|--------:|");
    Console.WriteLine($"| InferGetTop (per call) | {nf:F1} ms | {nm:F1} ms | {Mb(wFloat)} → {Mb(wFloat / 2)} |");
    Console.WriteLine("  (mixed re-quantises the masters on every call — the API's honest cost)");
}
finally
{
    try { Directory.Delete(tmp, recursive: true); } catch { }
}

// ── 3. GEMM 1024³: where the tensor cores live ───────────────────────────────
{
    const int n = 1024;
    var rng = new Random(3);
    var a = new float[n * n];
    var b = new float[n * n];
    for (int i = 0; i < a.Length; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
    for (int i = 0; i < b.Length; i++) b[i] = (float)(rng.NextDouble() * 2 - 1);

    var cpu = new CpuGemmProvider();
    double gf = MedianMs(() => cpu.MatMul(a, n, n, b, n));
    double gm = MedianMs(() => cpu.MatMulMixed(a, n, n, b, n));
    Console.WriteLine($"\n| GEMM {n}³ | time |");
    Console.WriteLine($"|---|-----:|");
    Console.WriteLine($"| CPU float32 | {gf:F0} ms |");
    Console.WriteLine($"| CPU mixed (rounding oracle) | {gm:F0} ms |");

    if (DeviceRuntime.CudaAvailable)
    {
        var cuda = new CudaGemmProvider();
        double gc = MedianMs(() => cuda.MatMulMixed(a, n, n, b, n));
        Console.WriteLine($"| CUDA mixed (tensor cores, incl. transfers) | {gc:F1} ms |");
    }
    else
    {
        Console.WriteLine("| CUDA mixed | (CUDA runtime not discoverable — skipped) |");
    }
}

Console.WriteLine("\nDone.");
return;

static double MedianMs(Action op, int reps = 5, int warmup = 2)
{
    for (int i = 0; i < warmup; i++) op();
    var times = new double[reps];
    for (int i = 0; i < reps; i++)
    {
        var sw = Stopwatch.StartNew();
        op();
        sw.Stop();
        times[i] = sw.Elapsed.TotalMilliseconds;
    }
    Array.Sort(times);
    return times[reps / 2];
}

static string Us(double ms) => $"{ms * 1000:F1} µs";
static string Mb(long bytes) => $"{bytes / (1024.0 * 1024.0):F1} MB";
