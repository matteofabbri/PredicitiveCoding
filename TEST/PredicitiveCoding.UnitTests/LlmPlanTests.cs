using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Eval;
using PredictiveCoding.LLM.Training;

namespace PredicitiveCoding.UnitTests;

// Phase 8 of PredictiveCoding.LLM/ROADMAP.md: latent planning — settle a plan
// vector on the prompt (the evidence), then decode under it. Plus the last
// Phase 0 leftover: the self-contained HTML report.
public class LlmPlanTests
{
    private static DecoderLmConfig Cfg(bool plan = false) =>
        new(13, 16, NumBlocks: 1, NumHeads: 2, FfDim: 32, MaxSeq: 16, UsePlan: plan);

    // ── Off means off; zero plan means unchanged ─────────────────────────────

    [Fact]
    public void PlanOn_ZeroInitialised_LeavesTheForwardUnchanged()
    {
        var off = new DecoderLM(Cfg(), new Random(3));
        var on  = new DecoderLM(Cfg(plan: true), new Random(3));   // plan is zero-init, not rng-drawn

        Assert.Equal(off.Parameters().Length + 1, on.Parameters().Length);

        var a = off.Forward([1, 2, 3]);
        var b = on.Forward([1, 2, 3]);
        for (int i = 0; i < a.Data.Length; i++)
            Assert.Equal(a.Data[i], b.Data[i]);                    // +0 everywhere ⇒ identical values
    }

    [Fact]
    public void PlanApi_Validates()
    {
        var off = new DecoderLM(Cfg(), new Random(1));
        Assert.Empty(off.GetPlan());
        Assert.Throws<InvalidOperationException>(() => off.SettlePlan([1, 2, 3], 5, 0.05f));
        Assert.Throws<InvalidOperationException>(() => off.SetPlan(new float[16]));

        var on = new DecoderLM(Cfg(plan: true), new Random(1));
        Assert.Throws<ArgumentException>(() => on.SetPlan(new float[3]));
        Assert.Throws<ArgumentException>(() => on.SettlePlan([7], 5, 0.05f));
    }

    // ── Settling: infer the latent from the evidence ─────────────────────────

    [Fact]
    public void SettlePlan_DescendsThePromptLoss_AndOnlyMovesThePlan()
    {
        var lm = new DecoderLM(Cfg(plan: true), new Random(5));
        var weightsBefore = lm.Parameters()[..^1].Select(p => (float[])p.Data.Clone()).ToArray();

        var trace = lm.SettlePlan([1, 5, 9, 2, 7, 3], steps: 30, learningRate: 0.05f);

        Assert.True(trace[^1] < trace[0], $"settling did not descend: {trace[0]:F3} → {trace[^1]:F3}");
        Assert.NotEqual(0f, lm.GetPlan().Select(MathF.Abs).Max());          // the plan actually moved

        var weightsAfter = lm.Parameters()[..^1].Select(p => p.Data).ToArray();
        for (int k = 0; k < weightsBefore.Length; k++)                       // …and nothing else did
            for (int i = 0; i < weightsBefore[k].Length; i++)
                Assert.Equal(weightsBefore[k][i], weightsAfter[k][i]);
    }

    [Fact]
    public void SettledPlan_ExplainsTheContinuationBetterThanNoise()
    {
        // Train a plan model on a cyclic corpus, then: settle the plan on a
        // prompt and score the continuation, against a random-noise plan.
        var lm  = new DecoderLM(Cfg(plan: true), new Random(6));
        var opt = new Adam();
        int[] toks = [1, 2, 3, 1, 2, 3, 1, 2], tgts = [2, 3, 1, 2, 3, 1, 2, 3];
        for (int it = 0; it < 150; it++)
        {
            lm.Loss(toks, tgts, ignoreIndex: -1, training: true).Backward();
            opt.Step(lm.Parameters(), 0.01f);
        }

        // Inputs [1,2,3,1,2]; next-token targets with the prompt half masked:
        // only the continuation (3→1, 1→2, 2→3) is scored.
        int[] input   = [1, 2, 3, 1, 2];
        int[] targets = [-1, -1, 1, 2, 3];

        // Gentle settling: a few small steps. An aggressive settle (lr 0.05,
        // 20 steps) demonstrably OVERFITS the 4-token prompt and wrecks the
        // continuation — the latent must fit the evidence, not memorise it.
        lm.SettlePlan([1, 2, 3, 1], steps: 10, learningRate: 0.01f);
        double settled = PerplexityEvaluator.Evaluate(lm, [(input, targets)]).NllPerToken;

        var noise = new float[16];
        var rng   = new Random(42);
        for (int i = 0; i < noise.Length; i++) noise[i] = (float)(rng.NextDouble() * 2 - 1);
        lm.SetPlan(noise);
        double random = PerplexityEvaluator.Evaluate(lm, [(input, targets)]).NllPerToken;

        Assert.True(settled <= random,
            $"a plan settled on the evidence should beat noise: settled {settled:F3} vs noise {random:F3}");
    }

    // ── Phase 0 leftover: the one-file report ────────────────────────────────

    [Fact]
    public void HtmlReport_IsSelfContained_LightThemed_AndCarriesTheNumbers()
    {
        var lm    = new DecoderLM(Cfg(), new Random(7));
        var probe = new ActivationProbe();
        lm.Forward([1, 5, 9, 2], training: false, probe);
        var ppl = PerplexityEvaluator.Evaluate(lm, [([1, 2, 3], [2, 3, 4])]);

        string html = HtmlReport.Render("LLM yardstick", ppl, probe);

        Assert.Contains("LLM yardstick", html);
        Assert.Contains(ppl.Perplexity.ToString("F2", System.Globalization.CultureInfo.InvariantCulture), html);
        Assert.Contains("Attention-sink share", html);
        Assert.Contains("background:#ffffff", html);               // light theme, always
        Assert.DoesNotContain("http://", html);                    // self-contained…
        Assert.DoesNotContain("https://", html);                   // …no external references

        string path = Path.Combine(Path.GetTempPath(), $"llmreport_{Guid.NewGuid():N}.html");
        try
        {
            HtmlReport.Save(path, "LLM yardstick", ppl, probe);
            Assert.True(new FileInfo(path).Length > 500);
        }
        finally { File.Delete(path); }
    }
}
