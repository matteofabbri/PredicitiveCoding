using PredictiveCoding.Cortex;
using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Autograd;

namespace PredicitiveCoding.UnitTests;

// The Cortex ROADMAP opening, realized: PredictiveLlm standing in as the real
// language area — Wernicke comprehends (embedding + surprise + online Listen),
// Broca produces (generation, and meaning-conditioned generation through the
// Phase-8 latent plan). The toy vector mode stays untouched (CortexLanguageTests).
public class CortexLlmLanguageTests
{
    private static PredictiveLlm TrainedLm(int seed = 21)
    {
        var llm = new PredictiveLlm(new DecoderLmConfig(
            260, Dim: 16, NumBlocks: 2, NumHeads: 2, FfDim: 32, MaxSeq: 64,
            BlockPattern: "TA", UsePlan: true), seed: seed);
        llm.Train(PredictiveCoding.LLM.Eval.EvalCorpus.Tiny,
            new TrainingOptions { SeqLen = 24, Stride = 12, Epochs = 5, BatchWindows = 6 });
        return llm;
    }

    // ── Wernicke: comprehension on the real model ────────────────────────────

    [Fact]
    public void Wernicke_Comprehends_ToTheSemanticLatent_Deterministically()
    {
        var wernicke = new WernickeArea(TrainedLm());

        Assert.True(wernicke.UsesLanguageModel);
        Assert.Equal(16, wernicke.SemanticDim);

        var a = wernicke.Comprehend("The brain is a prediction machine.");
        var b = wernicke.Comprehend("The brain is a prediction machine.");
        Assert.Equal(16, a.Length);
        for (int i = 0; i < a.Length; i++) Assert.Equal(a.Buffer[i], b.Buffer[i]);   // meaning is stable
    }

    [Fact]
    public void Wernicke_Surprise_SeparatesTheFamiliarFromTheAlien()
    {
        var wernicke = new WernickeArea(TrainedLm(seed: 23));

        float familiar = wernicke.Surprise("The brain is a prediction machine.");
        float alien    = wernicke.Surprise("zqxj vwpk qqzz xkcv jjwq");

        Assert.True(familiar < alien,
            $"a trained Wernicke should find corpus text less surprising: {familiar:F3} vs {alien:F3}");
    }

    [Fact]
    public void Wernicke_Listen_LearnsWhileHearing()
    {
        var wernicke = new WernickeArea(TrainedLm(seed: 25));
        const string novel = "surprise is the bill the world sends for a bad model";

        var report = wernicke.Listen(novel, steps: 2);
        Assert.True(report.SurpriseAfter < report.SurpriseBefore, report.ToString());
        Assert.True(wernicke.Surprise(novel) < report.SurpriseBefore);   // the area actually changed
    }

    // ── Broca: production on the real model ──────────────────────────────────

    [Fact]
    public void Broca_Produces_AndMeaningConditionsTheProduction()
    {
        var llm      = TrainedLm(seed: 27);
        var wernicke = new WernickeArea(llm);
        var broca    = new BrocaArea(llm);

        Assert.True(broca.UsesLanguageModel);

        string plain = broca.Produce("The brain is", maxNewTokens: 24);
        Assert.False(string.IsNullOrEmpty(plain));
        Assert.Equal(plain, broca.Produce("The brain is", maxNewTokens: 24));        // deterministic

        // The full comprehension → production loop: Wernicke's meaning becomes
        // Broca's latent plan. Gain 0 must equal unconditioned production…
        var meaning = wernicke.Comprehend("Learning is the slow art of being surprised less often.");
        Assert.Equal(plain, broca.ProduceFromMeaning(meaning, "The brain is", 24, planGain: 0f));

        // …and the plan is restored afterwards (stateless between calls).
        string again = broca.ProduceFromMeaning(meaning, "The brain is", 24, planGain: 0.2f);
        Assert.Equal(plain, broca.Produce("The brain is", maxNewTokens: 24));
        Assert.False(string.IsNullOrEmpty(again));
    }

    // ── Mode hygiene: the two worlds never blur ──────────────────────────────

    [Fact]
    public void WrongMode_FailsLoudly_AndPlanlessModelIsRefused()
    {
        var llm      = TrainedLm(seed: 29);
        var wernicke = new WernickeArea(llm);
        var broca    = new BrocaArea(llm);

        Assert.Throws<InvalidOperationException>(() =>
            wernicke.Comprehend([new PredictiveCoding.Core.DenseTensor<float>(4)], steps: 5));
        Assert.Throws<InvalidOperationException>(() =>
            broca.Produce(new PredictiveCoding.Core.DenseTensor<float>(16), steps: 5));

        var planless = new PredictiveLlm(new DecoderLmConfig(260, 16, 1, 2, 32, 32));
        var b2 = new BrocaArea(planless);
        Assert.Throws<InvalidOperationException>(() =>
            b2.ProduceFromMeaning(new PredictiveCoding.Core.DenseTensor<float>(16), "x", 4));
    }
}
