using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Training;

namespace PredicitiveCoding.UnitTests;

// Pro roadmap Phase 6: robust continual learning. Replay buffer, surprise-scaled
// lr, and the forgetting guardrail — with the ablation that proves replay works:
// absorbing a new domain wrecks the old one WITHOUT replay, but holds it WITH.
public class LlmProPhase6Tests
{
    // Two easily-separable "domains": repeated ASCII patterns the byte model can fit fast.
    private const string DomainA = "aaaa bbbb aaaa bbbb aaaa bbbb aaaa bbbb\n" +
                                   "aaaa bbbb aaaa bbbb aaaa bbbb aaaa bbbb";
    private const string DomainB = "1234 5678 1234 5678 1234 5678 1234 5678\n" +
                                   "1234 5678 1234 5678 1234 5678 1234 5678";

    private static PredictiveLlm TrainedOnA(int seed = 1)
    {
        var llm = PredictiveLlm.Create2026(dim: 24, blocks: 2, heads: 2, ffDim: 48, maxSeq: 48, seed: seed);
        llm.Train(string.Join('\n', Enumerable.Repeat(DomainA, 10)),
                  new TrainingOptions { SeqLen = 16, Stride = 4, Epochs = 25, BatchWindows = 8 });
        return llm;
    }

    // ── The ablation: replay prevents catastrophic forgetting ────────────────

    [Fact]
    public void Replay_HoldsTheOldDomain_WhileWithoutReplayItDegrades()
    {
        var opts = new ContinualOptions
        {
            SeqLen = 16, Stride = 4, BaseLr = 0.003f, Steps = 2,
            ReplayCapacity = 64, ReplayPerStep = 6, SurpriseEmaDecay = 0f,
        };

        // WITH replay: seed the buffer with domain-A windows, then absorb B.
        var withReplay = TrainedOnA(seed: 1);
        var crWith = new ContinualLearner(withReplay, opts, yardstickText: DomainA);
        double baseWith = crWith.BaselinePerplexity;
        crWith.Absorb(DomainA);                                   // fill the replay ring with A
        for (int i = 0; i < 12; i++) crWith.Absorb(DomainB);
        double aPplWith = crWith.MeasureYardstick();

        // WITHOUT replay: same everything, buffer disabled.
        var noReplay = TrainedOnA(seed: 1);
        var crNo = new ContinualLearner(noReplay, opts with { ReplayCapacity = 0, ReplayPerStep = 0 },
                                        yardstickText: DomainA);
        double baseNo = crNo.BaselinePerplexity;
        crNo.Absorb(DomainA);
        for (int i = 0; i < 12; i++) crNo.Absorb(DomainB);
        double aPplNo = crNo.MeasureYardstick();

        double driftWith = (aPplWith - baseWith) / baseWith;
        double driftNo   = (aPplNo   - baseNo)   / baseNo;

        // Replay must forget the old domain markedly LESS than no-replay.
        Assert.True(driftWith < driftNo,
            $"replay drift {driftWith:P0} should be below no-replay drift {driftNo:P0}");
        Assert.True(driftNo > 0.10,
            $"without replay the old domain should visibly degrade, drift was {driftNo:P0}");
    }

    // ── Surprise-proportional lr ─────────────────────────────────────────────

    [Fact]
    public void SurpriseScaling_SpendsMoreOnSurprisingText()
    {
        var llm  = TrainedOnA(seed: 2);
        var opts = new ContinualOptions
        {
            SeqLen = 16, Stride = 4, BaseLr = 0.002f,
            ReplayCapacity = 0, ReplayPerStep = 0,
            MinLrScale = 0.25f, MaxLrScale = 4f, SurpriseEmaDecay = 0.9f,
        };
        var cr = new ContinualLearner(llm, opts);

        cr.Absorb(DomainA);                                       // first absorb seeds the EMA at scale 1
        var familiar  = cr.Absorb(DomainA);                       // low surprise → below base
        var surprising = cr.Absorb(DomainB);                      // high surprise → above base

        Assert.True(surprising.EffectiveLr > familiar.EffectiveLr,
            $"surprising text ({surprising.EffectiveLr:E2}) should get a higher lr than familiar ({familiar.EffectiveLr:E2})");
        Assert.True(surprising.EffectiveLr <= opts.BaseLr * opts.MaxLrScale + 1e-9f);   // bounded
        Assert.True(familiar.EffectiveLr   >= opts.BaseLr * opts.MinLrScale - 1e-9f);
    }

    // ── Guardrail fires on drift ─────────────────────────────────────────────

    [Fact]
    public void Guardrail_FiresWhenTheYardstickDrifts()
    {
        var llm  = TrainedOnA(seed: 3);
        var opts = new ContinualOptions
        {
            SeqLen = 16, Stride = 4, BaseLr = 0.01f, Steps = 3,
            ReplayCapacity = 0, ReplayPerStep = 0,            // no replay ⇒ let it forget
            GuardrailEvery = 2, DriftThreshold = 0.10, SurpriseEmaDecay = 0f,
        };
        var cr = new ContinualLearner(llm, opts, yardstickText: DomainA);

        ContinualDrift? fired = null;
        cr.OnDrift = d => fired ??= d;

        for (int i = 0; i < 10; i++) cr.Absorb(DomainB);

        Assert.NotNull(fired);
        Assert.True(fired!.RelativeDrift > opts.DriftThreshold);
        Assert.Equal(cr.BaselinePerplexity, fired.BaselinePerplexity);
    }

    [Fact]
    public void Absorb_OnlyLearnsProvidedText_AndReportsSurprise()
    {
        var llm = TrainedOnA(seed: 4);
        var cr  = new ContinualLearner(llm, new ContinualOptions { SeqLen = 16, Stride = 4, Steps = 3 });

        var step = cr.Absorb(DomainA);
        Assert.True(step.SurpriseAfter <= step.SurpriseBefore + 1e-3f);   // a step on the text lowers its own loss
        Assert.Equal(1, cr.AbsorbCount);
    }
}
