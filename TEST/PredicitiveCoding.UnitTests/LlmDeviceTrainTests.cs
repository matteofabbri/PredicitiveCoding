using PredictiveCoding.Core;
using PredictiveCoding.Device;
using PredictiveCoding.Device.Interop;
using PredictiveCoding.Device.Kernels;
using PredictiveCoding.Device.Streams;
using PredictiveCoding.Device.Training;
using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Training;

namespace PredicitiveCoding.UnitTests;

// ROADMAP-Scale A1/A3: the device-resident training step. Every test is
// GPU-gated (returns immediately without CUDA) — on the RTX 4070 dev box they
// run for real: the CPU DecoderLM with the same RMSNorm+RoPE config is the
// reference twin the GPU must match.
public class LlmDeviceTrainTests
{
    private static bool NoGpu => !DeviceRuntime.CudaAvailable;

    // Small dense RMSNorm+RoPE config the trainer supports.
    private static DecoderLmConfig TinyCfg(int vocab = 211, int maxSeq = 40) =>
        new(vocab, Dim: 32, NumBlocks: 2, NumHeads: 4, FfDim: 64, MaxSeq: maxSeq,
            UseRmsNorm: true, UseRope: true);

    private static (int[] tokens, int[] targets) Window(DecoderLmConfig cfg, int seq, int seed)
    {
        var rng = new Random(seed);
        var tok = new int[seq]; var tgt = new int[seq];
        for (int i = 0; i < seq; i++) tok[i] = rng.Next(cfg.Vocab);
        for (int i = 0; i < seq; i++) tgt[i] = i + 1 < seq ? tok[i + 1] : rng.Next(cfg.Vocab);
        return (tok, tgt);
    }

    // Upload a float[] to the GPU through the public DeviceTensor API.
    private static DeviceTensor<float> Up(float[] data)
        => new DenseTensor<float>(data, data.Length).ToDevice(Device.Cuda());

    private static void SyncStream()
        => CudaStreamManager.Shared.Synchronize(CudaStreamManager.Shared.ThreadStream(0));

    // ── GEMM with transposes: the backward's workhorse ───────────────────────

    [Fact]
    public void MatMulDeviceEx_AllTransposeVariants_MatchCpu()
    {
        if (NoGpu) return;
        var rng = new Random(3);
        int m = 5, k = 7, n = 4;
        var stream = CudaStreamManager.Shared.ThreadStream(0);
        var gemm = new CudaGemmProvider();

        foreach (bool tA in (bool[])[false, true])
            foreach (bool tB in (bool[])[false, true])
            {
                // Physical shapes: A is [m,k] or [k,m]; B is [k,n] or [n,k].
                int aR = tA ? k : m, aC = tA ? m : k;
                int bR = tB ? n : k, bC = tB ? k : n;
                var A = new float[aR * aC]; var B = new float[bR * bC]; var C0 = new float[m * n];
                for (int i = 0; i < A.Length; i++) A[i] = (float)(rng.NextDouble() - 0.5);
                for (int i = 0; i < B.Length; i++) B[i] = (float)(rng.NextDouble() - 0.5);
                for (int i = 0; i < C0.Length; i++) C0[i] = (float)(rng.NextDouble() - 0.5);

                // CPU reference: C = op(A)·op(B) + C0.
                var expect = (float[])C0.Clone();
                for (int i = 0; i < m; i++)
                    for (int j = 0; j < n; j++)
                    {
                        float s = 0f;
                        for (int l = 0; l < k; l++)
                        {
                            float av = tA ? A[l * m + i] : A[i * k + l];
                            float bv = tB ? B[j * k + l] : B[l * n + j];
                            s += av * bv;
                        }
                        expect[i * n + j] += s;
                    }

                using var dA = Up(A); using var dB = Up(B); using var dC = Up(C0);
                gemm.MatMulDeviceEx(dA.DevicePtr, dB.DevicePtr, dC.DevicePtr, m, k, n, tA, tB, beta: 1f, stream);
                SyncStream();
                var got = dC.ToHost().Buffer;

                for (int i = 0; i < m * n; i++)
                    Assert.Equal(expect[i], got[i], 3);
            }
    }

    // ── Fused causal attention vs the autograd reference ─────────────────────

    [Fact]
    public void FusedAttention_ForwardAndBackward_MatchAutograd()
    {
        if (NoGpu) return;
        int seq = 9, heads = 3, headDim = 8;                 // power-of-two head dim
        float scale = 1f / MathF.Sqrt(headDim);
        var rng = new Random(11);

        // Per-head CPU tensors (autograd reference), packed device buffers.
        var q = new Tensor[heads]; var k = new Tensor[heads]; var v = new Tensor[heads];
        var packedQ = new float[seq * heads * headDim];
        var packedK = new float[seq * heads * headDim];
        var packedV = new float[seq * heads * headDim];
        for (int h = 0; h < heads; h++)
        {
            q[h] = new Tensor(seq, headDim); k[h] = new Tensor(seq, headDim); v[h] = new Tensor(seq, headDim);
            for (int i = 0; i < seq * headDim; i++)
            {
                q[h].Data[i] = (float)(rng.NextDouble() - 0.5);
                k[h].Data[i] = (float)(rng.NextDouble() - 0.5);
                v[h].Data[i] = (float)(rng.NextDouble() - 0.5);
                int row = i / headDim, col = i % headDim;
                packedQ[(row * heads + h) * headDim + col] = q[h].Data[i];
                packedK[(row * heads + h) * headDim + col] = k[h].Data[i];
                packedV[(row * heads + h) * headDim + col] = v[h].Data[i];
            }
        }

        // CPU reference: per head, softmaxCausal(q·kᵀ·scale)·v; loss = Σ_h mean(out_h ⊙ W_h).
        var coeff = new Tensor[heads];
        var outs  = new Tensor[heads];
        Tensor? loss = null;
        for (int h = 0; h < heads; h++)
        {
            outs[h] = Ag.MatMul(Ag.SoftmaxRowsCausal(Ag.Scale(Ag.MatMul(q[h], Ag.Transpose(k[h])), scale)), v[h]);
            coeff[h] = new Tensor(seq, headDim);
            for (int i = 0; i < coeff[h].Data.Length; i++) coeff[h].Data[i] = (float)(rng.NextDouble() - 0.5);
            var term = Ag.Mean(Ag.Mul(outs[h], coeff[h]));
            loss = loss is null ? term : Ag.Add(loss, term);
        }
        loss!.Backward();

        // Device: fused forward.
        var stream = CudaStreamManager.Shared.ThreadStream(0);
        IntPtr sh = stream.Handle;
        int nAll = seq * heads * headDim;
        using var dQ = Up(packedQ); using var dK = Up(packedK); using var dV = Up(packedV);
        using var dO = Up(new float[nAll]); using var dL = Up(new float[seq * heads]);
        DeviceTrainKernels.AttentionForward(sh, dQ.DevicePtr, dK.DevicePtr, dV.DevicePtr,
                                            dO.DevicePtr, dL.DevicePtr, seq, heads, headDim, scale);
        SyncStream();
        var gotO = dO.ToHost().Buffer;

        for (int h = 0; h < heads; h++)
            for (int row = 0; row < seq; row++)
                for (int col = 0; col < headDim; col++)
                    Assert.Equal(outs[h].Data[row * headDim + col],
                                 gotO[(row * heads + h) * headDim + col], 4);

        // Device: fused backward from the same upstream gradient dO = coeff/N.
        var packedDo = new float[nAll];
        int nEl = seq * headDim;
        for (int h = 0; h < heads; h++)
            for (int i = 0; i < nEl; i++)
            {
                int row = i / headDim, col = i % headDim;
                packedDo[(row * heads + h) * headDim + col] = coeff[h].Data[i] / nEl;
            }
        using var dDo = Up(packedDo);
        using var dDot = Up(new float[seq * heads]);
        using var dGq = Up(new float[nAll]); using var dGk = Up(new float[nAll]); using var dGv = Up(new float[nAll]);
        DeviceTrainKernels.AttentionDotDoO(sh, dDo.DevicePtr, dO.DevicePtr, dDot.DevicePtr, seq, heads, headDim);
        DeviceTrainKernels.AttentionBackward(sh, dQ.DevicePtr, dK.DevicePtr, dV.DevicePtr, dDo.DevicePtr,
                                             dL.DevicePtr, dDot.DevicePtr,
                                             dGq.DevicePtr, dGk.DevicePtr, dGv.DevicePtr,
                                             seq, heads, headDim, scale);
        SyncStream();
        var gq = dGq.ToHost().Buffer;
        var gk = dGk.ToHost().Buffer;
        var gv = dGv.ToHost().Buffer;

        for (int h = 0; h < heads; h++)
            for (int row = 0; row < seq; row++)
                for (int col = 0; col < headDim; col++)
                {
                    int cpu = row * headDim + col, dev = (row * heads + h) * headDim + col;
                    Assert.Equal(q[h].Grad[cpu], gq[dev], 4);
                    Assert.Equal(k[h].Grad[cpu], gk[dev], 4);
                    Assert.Equal(v[h].Grad[cpu], gv[dev], 4);
                }
    }

    // ── The whole wired step vs the CPU twin ─────────────────────────────────

    [Fact]
    public void DeviceTrainer_FirstLoss_MatchesTheCpuModel()
    {
        if (NoGpu) return;
        var cfg = TinyCfg();
        var model = new DecoderLM(cfg, new Random(7));
        var (tok, tgt) = Window(cfg, seq: 33, seed: 1);

        float cpuLoss = model.Loss(tok, tgt, ignoreIndex: -1, training: true).Data[0];

        using var trainer = new DeviceDecoderTrainer(model);
        float devLoss = trainer.Step(tok, tgt, lr: 0f);      // lr 0: read the loss, change nothing

        Assert.Equal(cpuLoss, devLoss, 3);
    }

    [Fact]
    public void DeviceTrainer_TracksCpuAdam_StepByStep()
    {
        if (NoGpu) return;
        var cfg = TinyCfg();
        var cpuModel = new DecoderLM(cfg, new Random(7));
        var devModel = new DecoderLM(cfg, new Random(7));    // identical weights
        var (tok, tgt) = Window(cfg, seq: 33, seed: 2);

        var adam = new Adam();                                // AdamW defaults, wd 0
        var cpuLosses = new float[8];
        for (int s = 0; s < 8; s++)
        {
            var loss = cpuModel.Loss(tok, tgt, ignoreIndex: -1, training: true);
            cpuLosses[s] = loss.Data[0];
            Adam.ZeroGrad(cpuModel.Parameters());
            loss.Backward();
            adam.Step(cpuModel.Parameters(), lr: 1e-3f);
        }

        using var trainer = new DeviceDecoderTrainer(devModel);
        var devLosses = new float[8];
        for (int s = 0; s < 8; s++) devLosses[s] = trainer.Step(tok, tgt, lr: 1e-3f);

        // Same trajectory (fp reduction order differs → small drift allowed)…
        for (int s = 0; s < 8; s++)
            Assert.True(MathF.Abs(cpuLosses[s] - devLosses[s]) < 5e-3f,
                $"step {s}: cpu {cpuLosses[s]:F5} vs device {devLosses[s]:F5}");
        // …and the model actually learns on both.
        Assert.True(devLosses[7] < devLosses[0], "device training did not reduce the loss");

        // The updated weights round-trip back and still agree with the CPU twin.
        trainer.ExportTo(devModel);
        var pc = cpuModel.Parameters(); var pd = devModel.Parameters();
        double worst = 0;
        for (int p = 0; p < pc.Length; p++)
            for (int i = 0; i < pc[p].Data.Length; i++)
                worst = Math.Max(worst, Math.Abs(pc[p].Data[i] - pd[p].Data[i]));
        Assert.True(worst < 5e-3, $"weights drifted {worst:E2} after 8 steps");
    }

    [Fact]
    public void DeviceTrainer_ImportExport_RoundTripsBitExact()
    {
        if (NoGpu) return;
        var cfg = TinyCfg();
        var src = new DecoderLM(cfg, new Random(21));
        var dst = new DecoderLM(cfg, new Random(99));        // different weights, same shapes

        using var trainer = new DeviceDecoderTrainer(src);
        trainer.ExportTo(dst);

        var ps = src.Parameters(); var pd = dst.Parameters();
        for (int p = 0; p < ps.Length; p++)
            Assert.Equal(ps[p].Data, pd[p].Data);            // upload→download is a pure copy
    }

    [Fact]
    public void DeviceTrainer_GradientAccumulation_AveragesWindows()
    {
        if (NoGpu) return;
        var cfg = TinyCfg();
        var model = new DecoderLM(cfg, new Random(5));
        var w1 = Window(cfg, seq: 24, seed: 3);
        var w2 = Window(cfg, seq: 24, seed: 4);

        using var trainer = new DeviceDecoderTrainer(model);
        float mean = trainer.Step([w1, w2], lr: 1e-3f);
        Assert.True(float.IsFinite(mean) && mean > 0f);

        // Next step still healthy (arena reuse, grads were reset).
        float next = trainer.Step([w1, w2], lr: 1e-3f);
        Assert.True(float.IsFinite(next));
        Assert.True(next < mean + 0.5f);
    }

    // ── A2: bf16 mixed compute on the tensor cores ───────────────────────────

    [Fact]
    public void Bf16Conversion_RoundTrips_WithinBf16Precision()
    {
        if (NoGpu) return;
        var rng = new Random(17);
        var src = new float[10_000];
        for (int i = 0; i < src.Length; i++) src[i] = (float)((rng.NextDouble() * 2 - 1) * Math.Pow(10, rng.Next(-8, 9)));

        var stream = CudaStreamManager.Shared.ThreadStream(0);
        using var dSrc = Up(src);
        using var dBf = Up(new float[src.Length / 2 + 1]);   // u16 storage in a float-sized buffer
        using var dBack = Up(new float[src.Length]);
        DeviceTrainKernels.FloatToBf16(stream.Handle, dSrc.DevicePtr, dBf.DevicePtr, src.Length);
        DeviceTrainKernels.Bf16ToFloat(stream.Handle, dBf.DevicePtr, dBack.DevicePtr, src.Length);
        SyncStream();
        var back = dBack.ToHost().Buffer;

        for (int i = 0; i < src.Length; i++)
        {
            float rel = Math.Abs(back[i] - src[i]) / Math.Max(1e-30f, Math.Abs(src[i]));
            Assert.True(rel <= 1f / 256f + 1e-6f, $"[{i}] {src[i]} → {back[i]} (rel {rel})");
        }
    }

    [Fact]
    public void MixedGemmBf16_MatchesFp32_WithinBf16Tolerance()
    {
        if (NoGpu) return;
        var rng = new Random(23);
        int m = 16, k = 48, n = 12;
        var A = new float[m * k]; var B = new float[k * n];
        for (int i = 0; i < A.Length; i++) A[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < B.Length; i++) B[i] = (float)(rng.NextDouble() - 0.5);

        var expect = new float[m * n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
            {
                float s = 0f;
                for (int l = 0; l < k; l++) s += A[i * k + l] * B[l * n + j];
                expect[i * n + j] = s;
            }

        var stream = CudaStreamManager.Shared.ThreadStream(0);
        var gemm = new CudaGemmProvider();
        using var dA = Up(A); using var dB = Up(B);
        using var dA16 = Up(new float[A.Length / 2 + 1]); using var dB16 = Up(new float[B.Length / 2 + 1]);
        using var dC = Up(new float[m * n]);
        DeviceTrainKernels.FloatToBf16(stream.Handle, dA.DevicePtr, dA16.DevicePtr, A.Length);
        DeviceTrainKernels.FloatToBf16(stream.Handle, dB.DevicePtr, dB16.DevicePtr, B.Length);
        gemm.MatMulDeviceExBf16(dA16.DevicePtr, dB16.DevicePtr, dC.DevicePtr, m, k, n, false, false, 0f, stream);
        SyncStream();
        var got = dC.ToHost().Buffer;

        for (int i = 0; i < m * n; i++)
            Assert.True(Math.Abs(got[i] - expect[i]) < 0.02f,
                $"[{i}] fp32 {expect[i]:F5} vs bf16 {got[i]:F5}");
    }

    [Fact]
    public void MixedTrainer_FirstLoss_CloseToTheCpuModel()
    {
        if (NoGpu) return;
        var cfg = TinyCfg();
        var model = new DecoderLM(cfg, new Random(7));
        var (tok, tgt) = Window(cfg, seq: 33, seed: 1);

        float cpuLoss = model.Loss(tok, tgt, ignoreIndex: -1, training: true).Data[0];

        using var trainer = new DeviceDecoderTrainer(model, mixedPrecision: true);
        Assert.True(trainer.MixedPrecision);
        float devLoss = trainer.Step(tok, tgt, lr: 0f);

        // bf16 GEMM rounding: ~0.4% per operand — allow a 2% relative band.
        Assert.True(MathF.Abs(devLoss - cpuLoss) / cpuLoss < 0.02f,
            $"cpu {cpuLoss:F4} vs bf16 device {devLoss:F4}");
    }

    [Fact]
    public void MixedTrainer_Converges_AndTracksTheFp32Trainer()
    {
        if (NoGpu) return;
        var cfg = TinyCfg();
        var m32 = new DecoderLM(cfg, new Random(7));
        var m16 = new DecoderLM(cfg, new Random(7));
        var (tok, tgt) = Window(cfg, seq: 33, seed: 2);

        using var t32 = new DeviceDecoderTrainer(m32);
        using var t16 = new DeviceDecoderTrainer(m16, mixedPrecision: true);
        var l32 = new float[10]; var l16 = new float[10];
        for (int s = 0; s < 10; s++) { l32[s] = t32.Step(tok, tgt, 1e-3f); l16[s] = t16.Step(tok, tgt, 1e-3f); }

        Assert.True(l16[9] < l16[0], "bf16 training did not reduce the loss");
        for (int s = 0; s < 10; s++)
            Assert.True(MathF.Abs(l32[s] - l16[s]) / l32[s] < 0.05f,
                $"step {s}: fp32 {l32[s]:F4} vs bf16 {l16[s]:F4}");
    }

    // ── C3: the device bootstrap loop ────────────────────────────────────────

    private static string BootstrapCorpus()
    {
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < 400; i++)
            sb.AppendLine("the brain is a prediction machine that minimizes surprise about the world it perceives.");
        return sb.ToString();
    }

    [Fact]
    public void Pretrainer_Bootstrap_TrainsEvaluatesAndCheckpoints()
    {
        if (NoGpu) return;
        string dir = Path.Combine(Path.GetTempPath(), $"pcboot_{Guid.NewGuid():N}");
        try
        {
            var llm = new PredictiveCoding.LLM.PredictiveLlm(
                TinyCfg(vocab: 260, maxSeq: 64), new PredictiveCoding.LLM.Tokenization.ByteTokenizer());
            var corpus = PredictiveCoding.LLM.Data.CorpusSource.FromText(BootstrapCorpus());

            float firstLoss = float.NaN;
            var report = DevicePretrainer.Pretrain(llm, corpus, new DevicePretrainOptions
            {
                Epochs = 2, SeqLen = 32, Stride = 32, MicroBatchWindows = 2,
                PeakLr = 3e-3f, WarmupSteps = 5, EvalEverySteps = 20, EvalWindows = 8,
                CheckpointDir = dir, CheckpointEverySteps = 50,
                OnProgress = p => { if (float.IsNaN(firstLoss)) firstLoss = p.TrainLoss; },
            });

            Assert.True(report.Steps > 10, $"only {report.Steps} steps ran");
            Assert.True(report.FinalTrainLoss < firstLoss,
                $"loss did not drop: {firstLoss:F3} → {report.FinalTrainLoss:F3}");
            Assert.True(float.IsFinite(report.FinalEvalLoss));
            Assert.True(File.Exists(Path.Combine(dir, "adam-device.bin")), "device Adam state missing");
            Assert.True(Directory.GetFiles(dir).Length >= 2, "model checkpoint missing");

            // Resume: a second run picks the Adam state up and keeps going.
            var again = DevicePretrainer.Pretrain(llm, corpus, new DevicePretrainOptions
            {
                Epochs = 1, SeqLen = 32, Stride = 32, MicroBatchWindows = 2,
                PeakLr = 1e-3f, WarmupSteps = 0, EvalEverySteps = 0,
                CheckpointDir = dir, CheckpointEverySteps = 1000,
            });
            Assert.True(again.Steps > 0);
            Assert.True(float.IsFinite(again.FinalTrainLoss));
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void Pretrainer_RealRun_KeepsBestCheckpoint_LogsAndReportsQuality()
    {
        if (NoGpu) return;
        string dir = Path.Combine(Path.GetTempPath(), $"pcrun_{Guid.NewGuid():N}");
        try
        {
            var llm = new PredictiveCoding.LLM.PredictiveLlm(
                TinyCfg(vocab: 260, maxSeq: 64), new PredictiveCoding.LLM.Tokenization.ByteTokenizer());
            var corpus = PredictiveCoding.LLM.Data.CorpusSource.FromText(BootstrapCorpus());
            string log = Path.Combine(dir, "train.jsonl");

            double? sawBits = null; string? sawSample = null;
            var report = DevicePretrainer.Pretrain(llm, corpus, new DevicePretrainOptions
            {
                Epochs = 3, SeqLen = 32, Stride = 32, MicroBatchWindows = 2, MaxSteps = 60,
                PeakLr = 3e-3f, WarmupSteps = 5,
                EvalEverySteps = 10, EvalWindows = 8,
                QualityEverySteps = 20, SamplePrompt = "the", SampleTokens = 16,
                CheckpointDir = dir, CheckpointEverySteps = 25, KeepBestCheckpoint = true,
                LogPath = log,
                OnProgress = p =>
                {
                    if (p.BitsPerByte is double b) sawBits = b;
                    if (p.Sample is not null) sawSample = p.Sample;
                },
            });

            Assert.Equal(60, report.Steps);                              // MaxSteps enforced
            Assert.True(File.Exists(Path.Combine(dir, "best")) || Directory.Exists(Path.Combine(dir, "best")),
                "best/ checkpoint missing");
            Assert.True(File.Exists(Path.Combine(dir, "best", "adam-device.bin")), "best Adam state missing");
            Assert.True(float.IsFinite(report.BestEvalLoss));
            Assert.True(report.BestEvalLoss <= report.FinalEvalLoss + 1e-4f, "best should be ≤ final");

            // Quality actually fired: a bits/byte number and a non-empty sample.
            Assert.NotNull(sawBits);
            Assert.True(double.IsFinite(sawBits!.Value) && sawBits.Value > 0);
            Assert.False(string.IsNullOrWhiteSpace(sawSample));
            Assert.NotNull(report.FinalBitsPerByte);

            // The log is JSONL, one object per line, growing.
            var lines = File.ReadAllLines(log);
            Assert.True(lines.Length >= 30, $"only {lines.Length} log rows");
            Assert.All(lines, l => { Assert.StartsWith("{", l); Assert.EndsWith("}", l); Assert.Contains("\"trainLoss\"", l); });

            // Resume appends to the same log rather than truncating it.
            int before = lines.Length;
            DevicePretrainer.Pretrain(llm, corpus, new DevicePretrainOptions
            {
                Epochs = 1, SeqLen = 32, Stride = 32, MicroBatchWindows = 2, MaxSteps = 10,
                PeakLr = 1e-3f, WarmupSteps = 0, EvalEverySteps = 0,
                CheckpointDir = dir, CheckpointEverySteps = 1000, LogPath = log,
            });
            Assert.True(File.ReadAllLines(log).Length > before, "resume did not append to the log");
        }
        finally { if (Directory.Exists(dir)) Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void EvaluateLoss_IsForwardOnly_WeightsUntouched()
    {
        if (NoGpu) return;
        var cfg = TinyCfg();
        var model = new DecoderLM(cfg, new Random(3));
        var probe = new DecoderLM(cfg, new Random(4));
        var (tok, tgt) = Window(cfg, seq: 24, seed: 6);

        using var trainer = new DeviceDecoderTrainer(model);
        trainer.ExportTo(probe);
        var before = probe.Parameters().Select(p => (float[])p.Data.Clone()).ToArray();

        float l1 = trainer.EvaluateLoss(tok, tgt);
        float l2 = trainer.EvaluateLoss(tok, tgt);
        Assert.Equal(l1, l2);                                 // deterministic, no drift

        trainer.ExportTo(probe);
        var after = probe.Parameters();
        for (int p = 0; p < after.Length; p++)
            Assert.Equal(before[p], after[p].Data);           // bit-identical weights
    }

    [Fact]
    public void Supports_RefusesWhatTheDevicePathCannotRun()
    {
        // Pure config logic — the false cases hold with or without a GPU.
        var good = TinyCfg();
        var layerNorm = good with { UseRmsNorm = false };
        var moe       = good with { NumExperts = 4 };
        var looped    = good with { NumLoops = 2 };
        var dropout   = good with { Dropout = 0.1f };

        if (DeviceRuntime.CudaAvailable) Assert.True(DeviceDecoderTrainer.Supports(good));
        Assert.False(DeviceDecoderTrainer.Supports(layerNorm));
        Assert.False(DeviceDecoderTrainer.Supports(moe));
        Assert.False(DeviceDecoderTrainer.Supports(looped));
        Assert.False(DeviceDecoderTrainer.Supports(dropout));
    }
}
