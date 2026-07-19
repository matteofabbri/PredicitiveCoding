using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Generation;

namespace PredicitiveCoding.UnitTests;

// Pro roadmap Phase 2: real decoding — temperature/top-k/top-p/penalties via
// the shared Sampler, stop sequences, and the multi-turn Chat surface with the
// self-feeding rule (learn from the user, never from the assistant).
public class LlmProPhase2Tests
{
    private static PredictiveLlm Tiny(int seed = 42, int maxSeq = 64)
        => new(new DecoderLmConfig(260, Dim: 16, NumBlocks: 2, NumHeads: 2, FfDim: 32, MaxSeq: maxSeq,
                                   BlockPattern: "TA", MtpWeight: 0.3f), seed: seed);

    private static PredictiveLlm TrainedOnCycle(int seed = 7, int maxSeq = 64)
    {
        var llm = Tiny(seed, maxSeq);
        llm.Train("abcabcabcabc\nabcabcabcabc", new TrainingOptions { SeqLen = 10, Stride = 5, Epochs = 8, BatchWindows = 4 });
        return llm;
    }

    // ── Sampling: deterministic knobs first ──────────────────────────────────

    [Fact]
    public void GreedySampling_And_TopK1_MatchTheDefaultGreedyPath()
    {
        var llm = TrainedOnCycle();
        string greedy = llm.Generate("abcabc", 12, new GenerationOptions { StopAtEos = false });

        string viaGreedyFlag = llm.Generate("abcabc", 12, new GenerationOptions
        {
            StopAtEos = false,
            Sampling  = new SamplingOptions { Greedy = true },
        });
        Assert.Equal(greedy, viaGreedyFlag);

        string viaTopK1 = llm.Generate("abcabc", 12, new GenerationOptions
        {
            StopAtEos = false,
            Sampling  = new SamplingOptions { Temperature = 1f, TopK = 1, Seed = 123 },
        });
        Assert.Equal(greedy, viaTopK1);                    // one survivor ⇒ argmax, any seed
    }

    [Fact]
    public void Sampling_IsReproduciblePerSeed_AndVariesAcrossSeeds()
    {
        var llm = Tiny(seed: 11);
        var optsA = new GenerationOptions { StopAtEos = false, Sampling = new SamplingOptions { Temperature = 1f, Seed = 1 } };
        var optsB = new GenerationOptions { StopAtEos = false, Sampling = new SamplingOptions { Temperature = 1f, Seed = 2 } };

        Assert.Equal(llm.Generate("hello", 24, optsA), llm.Generate("hello", 24, optsA));
        Assert.NotEqual(llm.Generate("hello", 24, optsA), llm.Generate("hello", 24, optsB));
    }

    [Fact]
    public void RepetitionPenalty_BreaksTheLearnedLoop()
    {
        var llm = TrainedOnCycle(seed: 9);
        var greedy = llm.Generate("abcabc", 16, new GenerationOptions { StopAtEos = false });

        var penalized = llm.Generate("abcabc", 16, new GenerationOptions
        {
            StopAtEos = false,
            Sampling  = new SamplingOptions { Temperature = 1f, TopK = 1, RepetitionPenalty = 4f, Seed = 5 },
        });

        Assert.NotEqual(greedy, penalized);                // the cycle cannot survive a 4× penalty
    }

    // ── Stop sequences ───────────────────────────────────────────────────────

    [Fact]
    public void StopSequences_CutGreedy_AndEndSamplingEarly()
    {
        var llm = TrainedOnCycle(seed: 13);

        string cut = llm.Generate("abcab", 16, new GenerationOptions { StopAtEos = false, StopSequences = ["c"] });
        Assert.DoesNotContain("c", cut);

        var sampled = llm.GenerateDetailed("abcab", 16, new GenerationOptions
        {
            StopAtEos     = false,
            StopSequences = ["c"],
            Sampling      = new SamplingOptions { Temperature = 1f, TopK = 1, Seed = 3 },
        });
        Assert.DoesNotContain("c", sampled.Text);
        Assert.True(sampled.Forwards < 16, $"early stop should save forwards, spent {sampled.Forwards}/16");
    }

    // ── The served path takes the same knobs ─────────────────────────────────

    [Fact]
    public void Server_TakesTheSameDecodingKnobs()
    {
        var llm = new PredictiveLlm(new DecoderLmConfig(260, 16, 2, 2, 32, 64, BlockPattern: "TA"));
        var server = llm.CreateServer();

        // Greedy via options ≡ the plain greedy overload.
        Assert.Equal(
            server.Generate("ab", 12, stopAtEos: false),
            server.Generate("ab", 12, new GenerationOptions { StopAtEos = false }));

        // Sampling: reproducible per seed.
        var opts = new GenerationOptions { StopAtEos = false, Sampling = new SamplingOptions { Temperature = 1f, Seed = 4 } };
        Assert.Equal(server.Generate("ab", 16, opts), server.Generate("ab", 16, opts));
    }

    // ── Chat ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Chat_KeepsHistory_AndRepliesStopAtRoleTags()
    {
        var llm  = TrainedOnCycle(seed: 15, maxSeq: 256);   // room for the role tags across turns
        var chat = llm.StartChat(system: "echo the pattern");

        string r1 = chat.Say("abc", maxNewTokens: 12, new GenerationOptions { StopAtEos = false });
        string r2 = chat.Say("abc again", maxNewTokens: 8, new GenerationOptions { StopAtEos = false });

        Assert.Equal(4, chat.History.Count);
        Assert.Equal(("user", "abc"), (chat.History[0].Role, chat.History[0].Content));
        Assert.Equal("assistant", chat.History[1].Role);
        Assert.Equal(r1, chat.History[1].Content);
        Assert.Equal(r2, chat.History[3].Content);
        Assert.All(new[] { r1, r2 }, r => Assert.DoesNotContain("<|", r));   // role tags never leak
    }

    [Fact]
    public void Chat_LearnFromUser_AbsorbsOnlyTheUserWords()
    {
        var llm  = Tiny(seed: 17, maxSeq: 256);
        var chat = llm.StartChat();
        const string userMsg = "the quick brown fox jumps over the lazy dog";

        double before = llm.Evaluate(userMsg).NllPerToken;
        chat.Say(userMsg, maxNewTokens: 4, new GenerationOptions { StopAtEos = false }, learnFromUser: true);
        double after = llm.Evaluate(userMsg).NllPerToken;

        Assert.True(after < before, $"the user's words should be less surprising now: {before:F3} → {after:F3}");
    }
}
