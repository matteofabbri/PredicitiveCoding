using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Generation;
using PredictiveCoding.LLM.Training;

namespace PredicitiveCoding.UnitTests;

// Phase 3 of PredictiveCoding.LLM/ROADMAP.md: multi-token prediction — an
// auxiliary head that predicts t+2 (the Nemotron 3 ablation), plus greedy
// self-speculative decoding that uses that head as its own drafter.
public class LlmMtpTests
{
    private static DecoderLmConfig Cfg(float mtpWeight = 0f, int horizon = 2) =>
        new(13, 8, NumBlocks: 1, NumHeads: 1, FfDim: 16, MaxSeq: 16, MtpWeight: mtpWeight, MtpHorizon: horizon);

    // ── Off means off ────────────────────────────────────────────────────────

    [Fact]
    public void MtpOff_ParameterListAndLoss_MatchVanilla()
    {
        var a = new DecoderLM(Cfg(), new Random(2));
        var b = new DecoderLM(Cfg(mtpWeight: 0f), new Random(2));

        Assert.Equal(a.Parameters().Length, b.Parameters().Length);
        Assert.Equal(
            a.Loss([1, 2, 3], [2, 3, 4], ignoreIndex: -1).Data[0],
            b.Loss([1, 2, 3], [2, 3, 4], ignoreIndex: -1).Data[0]);
    }

    [Fact]
    public void MtpOn_AddsExactlyOneHeadTensor()
    {
        var off = new DecoderLM(Cfg(), new Random(2));
        var on  = new DecoderLM(Cfg(mtpWeight: 0.5f), new Random(2));
        Assert.Equal(off.Parameters().Length + 1, on.Parameters().Length);

        var (_, mtpOff) = off.ForwardMtp([1, 2, 3]);
        var (_, mtpOn)  = on.ForwardMtp([1, 2, 3]);
        Assert.Null(mtpOff);
        Assert.NotNull(mtpOn);
    }

    // ── Target plumbing ──────────────────────────────────────────────────────

    [Fact]
    public void MtpTargets_ShiftIntoTheFutureAndMaskTheTail()
    {
        var lm2 = new DecoderLM(Cfg(mtpWeight: 0.5f, horizon: 2), new Random(1));
        Assert.Equal([3, 4, 5, -1], lm2.MtpTargets([2, 3, 4, 5], ignoreIndex: -1));

        var lm3 = new DecoderLM(Cfg(mtpWeight: 0.5f, horizon: 3), new Random(1));
        Assert.Equal([4, 5, -1, -1], lm3.MtpTargets([2, 3, 4, 5], ignoreIndex: -1));
    }

    // ── Both heads learn ─────────────────────────────────────────────────────

    [Fact]
    public void MtpTraining_ReducesBothHeadLosses()
    {
        var lm  = new DecoderLM(Cfg(mtpWeight: 0.5f), new Random(4));
        var opt = new Adam();
        int[] toks = [1, 2, 3, 4, 5], tgts = [2, 3, 4, 5, 6];

        (float main, float aux) Components()
        {
            var (main, mtp) = lm.ForwardMtp(toks);
            var mainCe = Ag.CrossEntropy(main, tgts, -1).Data[0];
            var auxCe  = Ag.CrossEntropy(mtp!, lm.MtpTargets(tgts, -1), -1).Data[0];
            return (mainCe, auxCe);
        }

        var before = Components();
        for (int it = 0; it < 150; it++)
        {
            lm.Loss(toks, tgts, ignoreIndex: -1, training: true).Backward();
            opt.Step(lm.Parameters(), 0.01f);
        }
        var after = Components();

        Assert.True(after.main < before.main, $"main head did not improve: {before.main:F3} → {after.main:F3}");
        Assert.True(after.aux  < before.aux,  $"t+2 head did not improve: {before.aux:F3} → {after.aux:F3}");
    }

    // ── Self-speculative decoding ────────────────────────────────────────────

    [Fact]
    public void SelfSpeculative_OutputIsExactlyGreedy_WithFewerForwards()
    {
        // A cyclic corpus the t+2 head can learn cold: 1 2 3 1 2 3 …
        var lm  = new DecoderLM(Cfg(mtpWeight: 0.5f), new Random(6));
        var opt = new Adam();
        int[] toks = [1, 2, 3, 1, 2, 3, 1, 2], tgts = [2, 3, 1, 2, 3, 1, 2, 3];
        for (int it = 0; it < 200; it++)
        {
            lm.Loss(toks, tgts, ignoreIndex: -1, training: true).Backward();
            opt.Step(lm.Parameters(), 0.01f);
        }

        int[] prompt = [1, 2, 3];
        var greedy = SelfSpeculative.Greedy(lm, prompt, maxNewTokens: 8);
        var spec   = SelfSpeculative.Generate(lm, prompt, maxNewTokens: 8);

        Assert.Equal(greedy, spec.Tokens);                 // identical by construction
        Assert.True(spec.Forwards <= 8, $"spent {spec.Forwards} forwards for 8 tokens");
        Assert.True(spec.Accepted > 0, "the trained t+2 head never got a draft accepted on a cyclic corpus");
        Assert.True(spec.Forwards < 8, $"acceptance should save forwards, spent {spec.Forwards}/8");
        Assert.InRange(spec.AcceptanceRate, 0f, 1f);
    }

    [Fact]
    public void SelfSpeculative_WithoutTheHead_FallsBackToPlainGreedy()
    {
        var lm = new DecoderLM(Cfg(), new Random(7));      // MtpWeight 0 ⇒ no drafter

        var greedy = SelfSpeculative.Greedy(lm, [1, 2], maxNewTokens: 5);
        var spec   = SelfSpeculative.Generate(lm, [1, 2], maxNewTokens: 5);

        Assert.Equal(greedy, spec.Tokens);
        Assert.Equal(0, spec.Drafted);
        Assert.Equal(5, spec.Forwards);                    // one forward per token, no savings
    }
}
