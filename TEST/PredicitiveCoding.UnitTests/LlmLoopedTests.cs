using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Eval;
using PredictiveCoding.LLM.Training;

namespace PredicitiveCoding.UnitTests;

// Phase 2 of PredictiveCoding.LLM/ROADMAP.md: looped (weight-tied) DecoderLM.
// Depth becomes iteration toward a fixed point — the discrete cousin of the
// core network's settling — and the loop can exit early at inference.
public class LlmLoopedTests
{
    // Per-block parameter count with 1 head: ln1(2) + qkvo(4) + ln2(2) + ff(4).
    private const int BlockParams = 12;

    // ── NumLoops = 1 is exactly the old model ────────────────────────────────

    [Fact]
    public void NumLoops1_IsTheDefault_SameParametersAndOutput()
    {
        var cfg = new DecoderLmConfig(31, 16, NumBlocks: 2, NumHeads: 2, FfDim: 32, MaxSeq: 8);
        var a = new DecoderLM(cfg, new Random(5));
        var b = new DecoderLM(cfg with { NumLoops = 1 }, new Random(5));

        Assert.Equal(a.Parameters().Length, b.Parameters().Length);   // no loop-bias tensors sneak in

        var la = a.Forward([1, 2, 3], training: false);
        var lb = b.Forward([1, 2, 3], training: false);
        for (int i = 0; i < la.Data.Length; i++)
            Assert.Equal(la.Data[i], lb.Data[i]);                     // bit-identical
    }

    [Fact]
    public void InvalidLoopConfig_Throws()
    {
        var cfg = new DecoderLmConfig(31, 16, 1, 1, 32, 8);
        Assert.Throws<ArgumentException>(() => new DecoderLM(cfg with { NumLoops = 0 }, new Random(1)));
        Assert.Throws<ArgumentException>(() => new DecoderLM(cfg with { LoopExitTau = -0.1f }, new Random(1)));
    }

    // ── The tied-reuse gradient: loop of 2 ≡ unrolled pair with shared weights ──

    [Fact]
    public void TiedLoop_Gradients_MatchTheUnrolledSum()
    {
        int[] tokens = [1, 2, 3], targets = [2, 3, 4];

        var looped   = new DecoderLM(new DecoderLmConfig(13, 8, NumBlocks: 1, NumHeads: 1, FfDim: 16, MaxSeq: 8, NumLoops: 2), new Random(3));
        var unrolled = new DecoderLM(new DecoderLmConfig(13, 8, NumBlocks: 2, NumHeads: 1, FfDim: 16, MaxSeq: 8), new Random(99));

        // Copy the looped weights into the unrolled twin: both unrolled blocks
        // get the SAME tied block, embedding/final-norm/head are shared too.
        // Looped layout:   [wemb][block:12][bias0][bias1][lnFG][lnFB][wout]
        // Unrolled layout: [wemb][block0:12][block1:12][lnFG][lnFB][wout]
        var lp = looped.Parameters();
        var up = unrolled.Parameters();
        Array.Copy(lp[0].Data, up[0].Data, lp[0].Data.Length);                       // wemb
        for (int i = 0; i < BlockParams; i++)
        {
            Array.Copy(lp[1 + i].Data, up[1 + i].Data, lp[1 + i].Data.Length);               // block0
            Array.Copy(lp[1 + i].Data, up[1 + BlockParams + i].Data, lp[1 + i].Data.Length); // block1 (tied copy)
        }
        for (int i = 0; i < 3; i++)                                                  // lnFG, lnFB, wout
            Array.Copy(lp[1 + BlockParams + 2 + i].Data, up[1 + 2 * BlockParams + i].Data, lp[1 + BlockParams + 2 + i].Data.Length);

        // Same forward values (the zero-initialised loop bias adds exactly 0)...
        var lossLooped   = looped.Loss(tokens, targets, ignoreIndex: -1);
        var lossUnrolled = unrolled.Loss(tokens, targets, ignoreIndex: -1);
        Assert.Equal(lossUnrolled.Data[0], lossLooped.Data[0], precision: 5);

        // ...and the tied gradient is the sum of the two unrolled uses.
        lossLooped.Backward();
        lossUnrolled.Backward();
        for (int i = 0; i < BlockParams; i++)
            for (int j = 0; j < lp[1 + i].Grad.Length; j++)
                Assert.Equal(up[1 + i].Grad[j] + up[1 + BlockParams + i].Grad[j], lp[1 + i].Grad[j], precision: 4);
    }

    // ── Training a tied loop works end to end ────────────────────────────────

    [Fact]
    public void LoopedModel_Training_ReducesLoss()
    {
        var lm  = new DecoderLM(new DecoderLmConfig(13, 8, NumBlocks: 1, NumHeads: 1, FfDim: 16, MaxSeq: 8, NumLoops: 3), new Random(4));
        var opt = new Adam();
        int[] toks = [1, 2, 3, 4], tgts = [2, 3, 4, 5];

        float first = lm.Loss(toks, tgts, ignoreIndex: -1).Data[0];
        for (int it = 0; it < 120; it++)
        {
            lm.Loss(toks, tgts, ignoreIndex: -1, training: true).Backward();
            opt.Step(lm.Parameters(), 0.01f);
        }
        float last = lm.Loss(toks, tgts, ignoreIndex: -1).Data[0];

        Assert.True(last < first * 0.5f, $"loss did not halve: {first:F3} → {last:F3}");
    }

    // ── Early exit: inference-only, tau = 0 disables ─────────────────────────

    [Fact]
    public void LoopEarlyExit_EvalOnly_TauZeroRunsTheWholeBudget()
    {
        var cfg   = new DecoderLmConfig(31, 16, NumBlocks: 1, NumHeads: 2, FfDim: 32, MaxSeq: 8, NumLoops: 6, LoopExitTau: 0.5f);
        var probe = new ActivationProbe();

        // Loose tau at eval: the residual stream stabilises and the exit fires.
        var lm = new DecoderLM(cfg, new Random(6));
        lm.Forward([1, 5, 9], training: false, probe);
        Assert.InRange(probe.LoopsUsed, 2, 5);                        // exited before the budget

        // Same weights, training pass: the exit must stay off.
        probe.Reset();
        lm.Forward([1, 5, 9], training: true, probe);
        Assert.Equal(6, probe.LoopsUsed);

        // Tau = 0: the whole budget runs at eval too.
        probe.Reset();
        var full = new DecoderLM(cfg with { LoopExitTau = 0f }, new Random(6));
        full.Forward([1, 5, 9], training: false, probe);
        Assert.Equal(6, probe.LoopsUsed);
    }

    [Fact]
    public void LoopedProbe_RecordsOneEntrySetPerPass()
    {
        var probe = new ActivationProbe();
        var lm = new DecoderLM(new DecoderLmConfig(31, 16, NumBlocks: 2, NumHeads: 2, FfDim: 32, MaxSeq: 8, NumLoops: 2), new Random(8));

        lm.Forward([1, 2, 3, 4], training: false, probe);

        Assert.Equal(2, probe.LoopsUsed);
        Assert.Equal(2 * 2, probe.Blocks.Count);                      // passes × blocks
        Assert.Equal(2 * 2 * 2, probe.Sinks.Count);                   // passes × blocks × heads
    }
}
