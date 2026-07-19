using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Training;

namespace PredicitiveCoding.UnitTests;

// Pro roadmap Phase 7: episodic memory, the hippocampal graft. Embed → Remember →
// Recall on the core MemoryBank; the two-session test proves facts stored in
// "session 1" are retrieved in "session 2" well above chance, and recall injects
// context into generation. The verdict's one sanctioned Brain/Cortex reuse.
//
// The model is trained ONCE (class fixture) — embeddings only separate the facts
// once the model has actually fit them, and training is the expensive part.
public sealed class Phase7Model
{
    // Each fact opens with a UNIQUE entity and ends with its "answer" word.
    public static readonly string[] Facts =
    [
        "Zephyria capital city is called Quon",
        "vibranium melting point is nine hundred degrees",
        "Belkorp admiral commands the northern fleet",
        "lanterns festival happens every autumn equinox",
        "Sable river flows east into crystal sea",
        "Vane professor discovered resonance of hollow metals",
    ];

    public PredictiveLlm Llm { get; }

    public Phase7Model()
    {
        Llm = PredictiveLlm.Create2026(dim: 32, blocks: 2, heads: 2, ffDim: 64, maxSeq: 64, seed: 1);
        Llm.Train(string.Join('\n', Enumerable.Repeat(string.Join('\n', Facts), 3)),
                  new TrainingOptions { SeqLen = 24, Stride = 4, Epochs = 40, BatchWindows = 8 });
    }

    /// <summary>A fresh memory holding every fact — tests stay independent of each other.</summary>
    public PredictiveLlm WithFreshMemory()
    {
        Llm.Memory = new EpisodicMemory(Llm.Config.Dim);
        foreach (var f in Facts) Llm.Remember(f);
        return Llm;
    }
}

public class LlmProPhase7Tests : IClassFixture<Phase7Model>, IDisposable
{
    private readonly Phase7Model _fixture;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"epi_{Guid.NewGuid():N}");

    public LlmProPhase7Tests(Phase7Model fixture) => _fixture = fixture;
    public void Dispose() { if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true); }

    private static string[] Facts => Phase7Model.Facts;

    // ── The two-session recall test ──────────────────────────────────────────

    [Fact]
    public void RememberThenRecall_RetrievesTheRightFact_AboveChance()
    {
        var llm = _fixture.WithFreshMemory();               // "session 1": store the facts
        Assert.Equal(Facts.Length, llm.Memory!.Count);

        // "session 2": query each fact by its SETUP (everything but the answer word);
        // the nearest memory should be the full fact, answer included.
        int hits = 0;
        foreach (var f in Facts)
        {
            string cue = string.Join(' ', f.Split(' ')[..^1]);
            var recalled = llm.Memory!.Recall(llm.Embed(cue), topK: 1);
            if (recalled.Count > 0 && recalled[0].text == f) hits++;
        }

        double hitRate = hits / (double)Facts.Length;
        double chance  = 1.0 / Facts.Length;                // ≈ 17%
        Assert.True(hitRate >= 0.66, $"recall hit-rate {hitRate:P0} should clearly beat chance {chance:P0}");
    }

    [Fact]
    public void MatchingCue_ScoresAboveAnUnrelatedQuery()
    {
        var llm = _fixture.WithFreshMemory();
        var matched   = llm.Memory!.Recall(llm.Embed("Zephyria capital city is called"), 1)[0];
        var unrelated = llm.Memory!.Recall(llm.Embed("qqqqq wwwww eeeee rrrrr"), 1)[0];
        Assert.Equal("Zephyria capital city is called Quon", matched.text);
        Assert.True(matched.similarity >= unrelated.similarity);
    }

    // ── Recall into generation ───────────────────────────────────────────────

    [Fact]
    public void UseMemory_RunsRecallIntoGeneration_WithinTheWindow()
    {
        var llm = _fixture.WithFreshMemory();
        var bare    = llm.GenerateDetailed("Zephyria capital city is", maxNewTokens: 1,
                        new GenerationOptions { UseMemory = false });
        var withCtx = llm.GenerateDetailed("Zephyria capital city is", maxNewTokens: 1,
                        new GenerationOptions { UseMemory = true, MemoryTopK = 2 });
        Assert.NotNull(bare.Text);
        Assert.NotNull(withCtx.Text);                       // recall path ran without overflowing the window
    }

    [Fact]
    public void Memory_SurvivesSaveLoad()
    {
        var llm = _fixture.WithFreshMemory();
        llm.Save(_dir);

        var reloaded = PredictiveLlm.Load(_dir);
        Assert.NotNull(reloaded.Memory);
        Assert.Equal(Facts.Length, reloaded.Memory!.Count);

        string cue = "Zephyria capital city is called";
        var a = llm.Memory!.Recall(llm.Embed(cue), 1)[0].text;
        var b = reloaded.Memory!.Recall(reloaded.Embed(cue), 1)[0].text;
        Assert.Equal(a, b);
    }

    // ── FIFO eviction keeps payloads aligned (controlled keys) ───────────────

    [Fact]
    public void EpisodicMemory_EvictsOldest_WhenOverCapacity()
    {
        var mem = new EpisodicMemory(keyDim: 4, capacity: 3);
        float[] K(int i) { var v = new float[4]; v[i] = 1f; return v; }   // orthogonal unit keys
        mem.Remember(K(0), "zero");
        mem.Remember(K(1), "one");
        mem.Remember(K(2), "two");
        mem.Remember(K(3), "three");                        // evicts "zero"

        Assert.Equal(3, mem.Count);
        Assert.Equal("three", mem.Recall(K(3), 1)[0].text);
        Assert.Equal("one",   mem.Recall(K(1), 1)[0].text);
        Assert.NotEqual("zero", mem.Recall(K(0), 1)[0].text);   // "zero" is gone
    }
}
