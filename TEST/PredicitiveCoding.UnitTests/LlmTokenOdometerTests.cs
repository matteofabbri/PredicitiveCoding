using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Eval;

namespace PredicitiveCoding.UnitTests;

// TokenOdometer: the model's lifetime token counter — logical tokens by
// activity on the product, compute tokens on the two stacks, persisted with
// the model like a real odometer.
public class LlmTokenOdometerTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"odo_{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true);
    }

    private static PredictiveLlm Tiny(int seed = 31)
        => new(new DecoderLmConfig(260, Dim: 16, NumBlocks: 2, NumHeads: 2, FfDim: 32, MaxSeq: 64,
                                   BlockPattern: "TA", MtpWeight: 0.3f), seed: seed);

    [Fact]
    public void FreshModel_StartsAtZero()
    {
        var llm = Tiny();
        Assert.Equal(0, llm.Odometer.Total);
        Assert.Equal(0, llm.Model.ForwardTokens);
        Assert.Contains("tokens: 0", llm.Odometer.ToString());
    }

    [Fact]
    public void Generate_CountsPromptAndEmittedTokens_Exactly()
    {
        var llm = Tiny();
        int promptTokens = llm.Tokenizer.Encode("hello world", addBos: true).Count;

        llm.Generate("hello world", maxNewTokens: 12, new GenerationOptions { StopAtEos = false });

        Assert.Equal(promptTokens, llm.Odometer.Prompted);
        Assert.Equal(12, llm.Odometer.Generated);
        // Compute side: the full-reforward path fed far more positions than 12.
        Assert.True(llm.Model.ForwardTokens > promptTokens + 12);
    }

    [Fact]
    public void Evaluate_And_Learn_CountTheirTokens_Exactly()
    {
        var llm = Tiny(seed: 33);

        var r = llm.Evaluate("The brain is a prediction machine.");
        Assert.Equal(r.Tokens, llm.Odometer.Evaluated);

        // "abc" is one whole-text window: BOS + 3 bytes + EOS = 5 ids → 4 inputs.
        long evaluatedBefore = llm.Odometer.Evaluated;
        llm.Learn("abc", steps: 3);
        Assert.Equal(4 * 3, llm.Odometer.Absorbed);                       // windows × steps
        Assert.Equal(evaluatedBefore + 4 * 2, llm.Odometer.Evaluated);    // before/after probes
    }

    [Fact]
    public void Train_CountsWindowsTimesEpochs_Exactly()
    {
        var llm  = Tiny(seed: 35);
        var opts = new TrainingOptions { SeqLen = 24, Stride = 12, Epochs = 3, BatchWindows = 6 };
        var report = llm.Train(EvalCorpus.Tiny, opts);

        // Streaming windows are fixed-length (SeqLen), so the count is exact.
        Assert.Equal((long)report.TrainWindows * 24 * 3, llm.Odometer.Trained);
        Assert.True(llm.Odometer.Evaluated > 0);                          // the before/after pair
    }

    [Fact]
    public void Odometer_SurvivesSaveLoad_AndKeepsCounting()
    {
        var llm = Tiny(seed: 37);
        llm.Generate("count me", maxNewTokens: 8, new GenerationOptions { StopAtEos = false });
        llm.Learn("the odometer keeps counting");
        var before = (llm.Odometer.Trained, llm.Odometer.Evaluated, llm.Odometer.Absorbed,
                      llm.Odometer.Prompted, llm.Odometer.Generated);

        llm.Save(_dir);
        var back = PredictiveLlm.Load(_dir);
        Assert.Equal(before, (back.Odometer.Trained, back.Odometer.Evaluated, back.Odometer.Absorbed,
                              back.Odometer.Prompted, back.Odometer.Generated));

        back.Generate("more", maxNewTokens: 4, new GenerationOptions { StopAtEos = false });
        Assert.Equal(before.Generated + 4, back.Odometer.Generated);      // continues, not restarts

        back.Odometer.Reset();
        Assert.Equal(0, back.Odometer.Total);
    }

    [Fact]
    public void ServedModel_CountsOneTokenPerStep()
    {
        var llm    = Tiny(seed: 39);
        var server = llm.CreateServer();
        int promptTokens = llm.Tokenizer.Encode("ab", addBos: true).Count;

        server.Generate("ab", maxNewTokens: 10, stopAtEos: false);

        // Incremental decoding: prompt feed + one step per emitted token — O(n), not O(n²).
        Assert.Equal(promptTokens + 10, server.Model.ForwardTokens);
    }
}
