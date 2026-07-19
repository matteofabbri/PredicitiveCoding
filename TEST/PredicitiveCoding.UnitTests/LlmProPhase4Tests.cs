using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Data;
using PredictiveCoding.LLM.Eval;
using PredictiveCoding.LLM.Tokenization;

namespace PredicitiveCoding.UnitTests;

// Pro roadmap Phase 4: the data pipeline — CorpusSource (files/folders/globs,
// weighted mixing, dedup + length hygiene, byte/line accounting) and the
// content-hashed held-out split that keeps perplexity comparable across runs.
public class LlmProPhase4Tests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), $"corpus_{Guid.NewGuid():N}");

    public LlmProPhase4Tests() => Directory.CreateDirectory(_dir);
    public void Dispose() { if (Directory.Exists(_dir)) Directory.Delete(_dir, recursive: true); }

    // ── Leaf sources + accounting ────────────────────────────────────────────

    [Fact]
    public void FromFolder_ReadsEveryFile_InDeterministicOrder()
    {
        File.WriteAllText(Path.Combine(_dir, "b.txt"), "beta1\nbeta2");
        File.WriteAllText(Path.Combine(_dir, "a.txt"), "alpha1\nalpha2");
        File.WriteAllText(Path.Combine(_dir, "skip.md"), "ignored");

        var src   = CorpusSource.FromFolder(_dir, "*.txt");
        var lines = src.Lines().ToList();

        Assert.Equal(["alpha1", "alpha2", "beta1", "beta2"], lines);   // a.txt before b.txt, .md excluded
        Assert.Equal(4, src.Stats.Lines);
        Assert.True(src.Stats.Bytes > 0);
    }

    [Fact]
    public void Glob_Recursive_FindsNestedFiles()
    {
        Directory.CreateDirectory(Path.Combine(_dir, "sub"));
        File.WriteAllText(Path.Combine(_dir, "top.txt"), "top");
        File.WriteAllText(Path.Combine(_dir, "sub", "deep.txt"), "deep");

        var lines = CorpusSource.FromGlob(_dir, "**.txt").Lines().OrderBy(x => x).ToList();
        Assert.Equal(["deep", "top"], lines);
    }

    // ── Hygiene ─────────────────────────────────────────────────────────────

    [Fact]
    public void Deduplicated_DropsRepeatLines_KeepingFirstOrder()
    {
        var src = CorpusSource.FromText("a\nb\na\nc\nb\nc").Deduplicated();
        Assert.Equal(["a", "b", "c"], src.Lines().ToList());
    }

    [Fact]
    public void LengthFilter_TrimsShortAndLongLines()
    {
        var src = CorpusSource.FromText("\nok\nx\ntoolongline").WithLengthFilter(minChars: 2, maxChars: 5);
        Assert.Equal(["ok"], src.Lines().ToList());   // "" and "x" too short, "toolongline" too long
    }

    // ── Weighted mixing ──────────────────────────────────────────────────────

    [Fact]
    public void Mix_InterleavesProportionallyToWeights()
    {
        // 60 A-lines vs 20 B-lines at 3:1 → ≈ equal exhaustion; check the ratio
        // over the first chunk stays near 3:1 (deterministic, no RNG).
        var a = CorpusSource.FromText(string.Join('\n', Enumerable.Range(0, 60).Select(_ => "A")));
        var b = CorpusSource.FromText(string.Join('\n', Enumerable.Range(0, 60).Select(_ => "B")));
        var mixed = CorpusSource.Mix((a, 3.0), (b, 1.0)).Lines().Take(40).ToList();

        int aCount = mixed.Count(l => l == "A");
        int bCount = mixed.Count(l => l == "B");
        Assert.Equal(40, aCount + bCount);
        Assert.InRange(aCount / (double)bCount, 2.5, 3.5);   // ≈ 3:1
    }

    [Fact]
    public void Mix_RejectsNonPositiveWeights()
    {
        var s = CorpusSource.FromText("x");
        Assert.Throws<ArgumentOutOfRangeException>(() => CorpusSource.Mix((s, 0.0)));
    }

    // ── Content-hashed split: position-independent, complementary ────────────

    [Fact]
    public void HashedSplit_IsComplementary_AndReorderStable()
    {
        var tok = new ByteTokenizer();
        var windows = new StreamingTextDataset(
            [.. Enumerable.Range(0, 200).Select(i => $"line number {i} with words")],
            tok, seqLen: 8, stride: 4).Windows().ToList();

        var held  = PerplexityEvaluator.HeldOutHashed(windows, every: 5).ToList();
        var train = PerplexityEvaluator.TrainingSplitHashed(windows, every: 5).ToList();

        Assert.Equal(windows.Count, held.Count + train.Count);            // exact partition
        Assert.True(held.Count > 0 && train.Count > 0);

        // Reversing the stream must not move a window between splits (position-independent).
        var reversed = Enumerable.Reverse(windows).ToList();
        var heldRev  = PerplexityEvaluator.HeldOutHashed(reversed, every: 5).ToList();
        Assert.Equal(held.Count, heldRev.Count);
    }

    // ── End-to-end: train from a mixed, deduped corpus ───────────────────────

    [Fact]
    public void Train_FromCorpusSource_ReducesLoss()
    {
        File.WriteAllText(Path.Combine(_dir, "domain.txt"),
            string.Join('\n', Enumerable.Repeat("the quick brown fox jumps over the lazy dog", 20)));
        File.WriteAllText(Path.Combine(_dir, "yard.txt"),
            string.Join('\n', Enumerable.Repeat("pack my box with five dozen liquor jugs", 20)));

        var source = CorpusSource.Mix(
            (CorpusSource.FromFile(Path.Combine(_dir, "domain.txt")), 0.8),
            (CorpusSource.FromFile(Path.Combine(_dir, "yard.txt")), 0.2))
            .Deduplicated()                    // collapses the repeats
            .WithLengthFilter(minChars: 5);

        var llm    = PredictiveLlm.Create2026(dim: 16, blocks: 2, heads: 2, ffDim: 32, maxSeq: 48);
        var report = llm.Train(source, new TrainingOptions { SeqLen = 16, Stride = 8, Epochs = 10, BatchWindows = 4 });

        Assert.True(report.LastLoss < report.FirstLoss,
            $"corpus-source training did not reduce loss: {report.FirstLoss:F3} → {report.LastLoss:F3}");
        Assert.True(source.Stats.Lines > 0);
    }
}
