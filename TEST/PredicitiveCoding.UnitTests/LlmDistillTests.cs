using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Generation;
using PredictiveCoding.LLM.Tokenization;
using PredictiveCoding.LLM.Training;

namespace PredicitiveCoding.UnitTests;

// Phase 7 of PredictiveCoding.LLM/ROADMAP.md: on-policy distillation with a
// canned teacher — no network anywhere near CI (the LM Studio hookup is a
// one-line delegate in app code).
public class LlmDistillTests : IDisposable
{
    private readonly string _cacheDir = Path.Combine(Path.GetTempPath(), $"distill_{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_cacheDir)) Directory.Delete(_cacheDir, recursive: true);
    }

    private static DecoderLM Student(int seed = 5) =>
        new(new DecoderLmConfig(260, 16, NumBlocks: 1, NumHeads: 2, FfDim: 32, MaxSeq: 64), new Random(seed));

    // ── Dataset cache ────────────────────────────────────────────────────────

    [Fact]
    public void BuildDataset_AsksTheTeacherOncePerPrompt_ThenReplaysFromDisk()
    {
        int calls = 0;
        string Teacher(string p) { calls++; return $"answer to {p}"; }

        var first = LmDistiller.BuildDataset(["ping", "pong"], Teacher, _cacheDir);
        Assert.Equal(2, calls);
        Assert.Equal("answer to ping", first[0].Completion);

        var second = LmDistiller.BuildDataset(["ping", "pong"], Teacher, _cacheDir);
        Assert.Equal(2, calls);                                   // cache hit: teacher untouched
        Assert.Equal(first[0].Completion, second[0].Completion);

        LmDistiller.BuildDataset(["new one"], Teacher, _cacheDir);
        Assert.Equal(3, calls);                                   // unseen prompt: one more call
    }

    // ── Masking: the context conditions, the completion trains ───────────────

    [Fact]
    public void BuildExample_MasksTheContext_TrainsCompletionPlusEos()
    {
        var tok = new ByteTokenizer();
        var (tokens, targets) = LmDistiller.BuildExample(tok, "AB", "xy", maxSeq: 64);

        var ctxIds  = tok.Encode("AB", addBos: true);
        var compIds = tok.Encode("xy", addEos: true);
        Assert.Equal(ctxIds.Count + compIds.Count - 1, tokens.Length);

        // Everything whose target is still context is masked…
        for (int i = 0; i + 1 < ctxIds.Count; i++) Assert.Equal(-1, targets[i]);
        // …and the scorable targets are exactly the completion + EOS.
        Assert.Equal(compIds, targets.Where(t => t != -1));
        Assert.Equal(tok.Eos, targets[^1]);
    }

    // ── Offline KD learns the teacher's answer ───────────────────────────────

    [Fact]
    public void OfflineDistillation_ReducesTheMaskedLoss()
    {
        var tok     = new ByteTokenizer();
        var student = Student();
        var opt     = new Adam();
        var data    = LmDistiller.BuildDataset(["count"], _ => "1 2 3", _cacheDir);

        float first = LmDistiller.Step(student, tok, data, opt, learningRate: 0.01f, clipNorm: 1f);
        float last  = 0f;
        for (int it = 0; it < 40; it++)
            last = LmDistiller.Step(student, tok, data, opt, learningRate: 0.01f, clipNorm: 1f);

        Assert.True(last < first * 0.5f, $"distillation did not bite: {first:F3} → {last:F3}");
    }

    // ── On-policy: the teacher corrects the student's own draft ──────────────

    [Fact]
    public void OnPolicyRound_TeacherSeesTheDraft_AndRoundsConverge()
    {
        var tok     = new ByteTokenizer();
        var student = Student(seed: 9);
        var opt     = new Adam();

        string? seenDraft = null;
        string Teacher(string prompt, string draft)
        {
            seenDraft = draft;
            return "the corrected answer";
        }

        float first = LmDistiller.OnPolicyRound(student, tok, ["fix me"], Teacher, opt, 0.01f, draftTokens: 6, clipNorm: 1f);
        Assert.NotNull(seenDraft);                                // the teacher really got the student's trajectory
        Assert.True(float.IsFinite(first) && first > 0f);

        float last = first;
        for (int round = 0; round < 15; round++)
            last = LmDistiller.OnPolicyRound(student, tok, ["fix me"], Teacher, opt, 0.01f, draftTokens: 6, clipNorm: 1f);

        Assert.True(last < first, $"on-policy rounds did not converge: {first:F3} → {last:F3}");
    }
}
