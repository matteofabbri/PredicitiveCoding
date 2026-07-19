using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Eval;
using PredictiveCoding.LLM.Generation;

namespace PredicitiveCoding.UnitTests;

// The finished product: PredictiveLlm composes every roadmap mechanism behind
// one class. These tests drive it end to end — create, train, generate,
// evaluate, distill, save/load, report.
public class PredictiveLlmTests : IDisposable
{
    private readonly string _tempRoot = Path.Combine(Path.GetTempPath(), $"pcllm_{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_tempRoot)) Directory.Delete(_tempRoot, recursive: true);
    }

    /// <summary>Small but with EVERY feature on — the product config, shrunk for CI.</summary>
    private static PredictiveLlm TinyFullFeature(int seed = 42)
        => new(new DecoderLmConfig(
                   260, Dim: 16, NumBlocks: 2, NumHeads: 2, FfDim: 32, MaxSeq: 48,
                   NumLoops: 2, LoopExitTau: 0.05f, MtpWeight: 0.3f,
                   BlockPattern: "TA", NumExperts: 2, MoeTopK: 1, UsePlan: true),
               seed: seed);

    // ── The preset really lights everything ──────────────────────────────────

    [Fact]
    public void Create2026_LightsEveryRoadmapFeature()
    {
        var llm = PredictiveLlm.Create2026();

        Assert.True(llm.Config.NumLoops > 1);                       // looped depth
        Assert.True(llm.Config.MtpWeight > 0);                      // t+2 head
        Assert.Contains('T', llm.Config.BlockPattern!);             // temporal-state blocks
        Assert.Contains('A', llm.Config.BlockPattern!);             // …interleaved with attention
        Assert.True(llm.Config.NumExperts > 1);                     // sparse experts
        Assert.True(llm.Config.UsePlan);                            // latent plan
        Assert.True(llm.ActiveParameterCountPerToken < llm.ParameterCount);   // MoE sparsity is real
    }

    [Fact]
    public void Constructor_RejectsVocabTokenizerMismatch()
    {
        Assert.Throws<ArgumentException>(() =>
            new PredictiveLlm(new DecoderLmConfig(99, 16, 1, 2, 32, 16)));
    }

    // ── Train end to end: the perplexity yardstick moves ─────────────────────

    [Fact]
    public void Train_OnTheYardstickCorpus_ImprovesPerplexity()
    {
        var llm = TinyFullFeature();
        var report = llm.Train(EvalCorpus.Tiny, new TrainingOptions
        {
            SeqLen = 24, Stride = 12, Epochs = 5, BatchWindows = 6,
        });

        Assert.True(report.PerplexityAfter < report.PerplexityBefore,
            $"training did not improve the model: {report}");
        Assert.True(report.LastLoss < report.FirstLoss, report.ToString());
        Assert.True(report.TrainWindows > 0 && report.EvalWindows > 0);
    }

    // ── Generate: speculative ≡ greedy, plan settling contained ──────────────

    [Fact]
    public void Generate_IsExactlyGreedy_WithSpeculativeSavings()
    {
        var llm = TinyFullFeature(seed: 7);
        llm.Train("abcabcabcabc\nabcabcabcabc", new TrainingOptions { SeqLen = 10, Stride = 5, Epochs = 8, BatchWindows = 4 });

        var detailed = llm.GenerateDetailed("abcabc", maxNewTokens: 8, new GenerationOptions { StopAtEos = false });

        // Reference: plain greedy on the same model must give the same text.
        var ctx = llm.Tokenizer.Encode("abcabc", addBos: true);
        var greedy = SelfSpeculative.Greedy(llm.Model, ctx, 8);
        Assert.Equal(llm.Tokenizer.Decode(greedy[ctx.Count..]), detailed.Text);

        Assert.InRange(detailed.AcceptanceRate, 0f, 1f);
        Assert.True(detailed.Forwards <= 8, $"spent {detailed.Forwards} forwards for 8 tokens");
        Assert.Equal(detailed.Text, llm.Generate("abcabc", 8, new GenerationOptions { StopAtEos = false }));
    }

    [Fact]
    public void Generate_WithPlanSettling_RestoresThePlanAfterwards()
    {
        var llm = TinyFullFeature(seed: 9);
        var before = llm.Model.GetPlan();

        var text = llm.Generate("hello world", maxNewTokens: 4,
            new GenerationOptions { SettlePlanSteps = 5, SettlePlanLr = 0.01f, StopAtEos = false });

        Assert.NotNull(text);
        Assert.Equal(before, llm.Model.GetPlan());                  // stateless between calls
        Assert.Throws<ArgumentException>(() => llm.Generate(new string('x', 200)));   // prompt > window
    }

    // ── Save / Load: bit-faithful round trip ─────────────────────────────────

    [Fact]
    public void SaveLoad_RoundTripsConfigAndWeights_Bitwise()
    {
        var llm = TinyFullFeature(seed: 5);
        llm.Train(EvalCorpus.Tiny, new TrainingOptions { SeqLen = 24, Stride = 12, Epochs = 2, BatchWindows = 6 });
        double ppl = llm.Evaluate(EvalCorpus.Tiny).Perplexity;

        string dir = Path.Combine(_tempRoot, "model");
        llm.Save(dir);
        var back = PredictiveLlm.Load(dir);

        Assert.Equal(llm.Config, back.Config);                      // record equality: every knob survives
        Assert.Equal(ppl, back.Evaluate(EvalCorpus.Tiny).Perplexity);   // same weights bit for bit
        Assert.Equal(llm.Generate("The brain", 6), back.Generate("The brain", 6));
    }

    // ── Distillation through the facade ──────────────────────────────────────

    [Fact]
    public void DistillFrom_ACannedTeacher_LearnsIt()
    {
        var llm = TinyFullFeature(seed: 11);
        string cache = Path.Combine(_tempRoot, "distill");

        float first = llm.DistillFrom(_ => "1 2 3", ["count"], steps: 1, cacheDir: cache);
        float last  = llm.DistillFrom(_ => "1 2 3", ["count"], steps: 40, cacheDir: cache);

        Assert.True(last < first * 0.6f, $"distillation did not bite: {first:F3} → {last:F3}");

        string? seenDraft = null;
        float onPolicy = llm.DistillOnPolicy((p, draft) => { seenDraft = draft; return "the answer"; }, ["fix"], draftTokens: 4);
        Assert.NotNull(seenDraft);                                  // the teacher saw the model's own draft
        Assert.True(float.IsFinite(onPolicy));
    }

    // ── Report: the observatory in one file ──────────────────────────────────

    [Fact]
    public void Report_WritesTheSelfContainedObservatory()
    {
        var llm  = TinyFullFeature(seed: 3);
        string path = Path.Combine(_tempRoot, "report.html");
        Directory.CreateDirectory(_tempRoot);

        Assert.Equal(path, llm.Report(path));
        string html = File.ReadAllText(path);
        Assert.Contains("Perplexity", html);
        Assert.Contains("Loop passes", html);                       // the looped forward is visible
        Assert.DoesNotContain("https://", html);                    // self-contained, as always
    }
}
