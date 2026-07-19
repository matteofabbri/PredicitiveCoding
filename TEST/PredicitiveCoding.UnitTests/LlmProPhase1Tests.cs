using System.Diagnostics;
using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Serving;

namespace PredicitiveCoding.UnitTests;

// Pro roadmap Phase 1: the serving path — train on the DecoderLM, serve on the
// GptModel cache stack. The exporter maps every supported mechanism (hybrid
// T/A blocks, MoE experts, absorbed plan) and fails loudly outside the subset.
public class LlmProPhase1Tests
{
    private static PredictiveLlm ServableFullFeature(int seed = 42)
        => new(new DecoderLmConfig(
                   260, Dim: 16, NumBlocks: 2, NumHeads: 2, FfDim: 32, MaxSeq: 48,
                   NumLoops: 1,                       // the declared serving subset
                   MtpWeight: 0.3f, BlockPattern: "TA",
                   NumExperts: 2, MoeTopK: 1, UsePlan: true),
               seed: seed);

    // ── Numerical parity: the exported model IS the trained model ────────────

    [Fact]
    public void Export_LogitsMatchTheTrainableForward()
    {
        var llm = ServableFullFeature();
        llm.Train("abcabcabcabc\nabcabcabcabc", new TrainingOptions { SeqLen = 10, Stride = 5, Epochs = 4, BatchWindows = 4 });

        int[] tokens = [97, 98, 99, 97, 98, 99];               // "abcabc" as bytes
        var reference = llm.Model.Forward(tokens);             // [seq, vocab], full re-forward
        int vocab = reference.Cols;

        var served = GptExporter.Export(llm.Model);
        var cache  = served.CreateCache();

        float maxDiff = 0f;
        for (int p = 0; p < tokens.Length; p++)
        {
            var logits = served.Step(tokens[p], p, cache);     // incremental, KV cache
            for (int v = 0; v < vocab; v++)
                maxDiff = MathF.Max(maxDiff, MathF.Abs(logits.Buffer[v] - reference.Data[p * vocab + v]));
        }

        Assert.True(maxDiff < 1e-3f, $"served logits drifted from the trainable forward: max |Δ| = {maxDiff:E2}");
    }

    // ── Text parity: same greedy continuation on both paths ──────────────────

    [Fact]
    public void Server_GeneratesTheSameGreedyTextAsTheTrainableModel()
    {
        var llm = ServableFullFeature(seed: 7);
        llm.Train("abcabcabcabc\nabcabcabcabc", new TrainingOptions { SeqLen = 10, Stride = 5, Epochs = 8, BatchWindows = 4 });

        var server = llm.CreateServer();
        string reforward = llm.Generate("abcabc", 16, new GenerationOptions { StopAtEos = false });
        string servedTxt = server.Generate("abcabc", 16, stopAtEos: false);

        Assert.Equal(reforward, servedTxt);
    }

    // ── The declared subset: loops are refused loudly ────────────────────────

    [Fact]
    public void Export_LoopedModel_ThrowsAClearError()
    {
        var looped = PredictiveLlm.Create2026(dim: 16, blocks: 2, heads: 2, ffDim: 32, maxSeq: 32);   // preset has NumLoops = 2
        var ex = Assert.Throws<NotSupportedException>(() => looped.CreateServer());
        Assert.Contains("NumLoops", ex.Message);
    }

    // ── The point of it all: O(n) beats O(n²) on the wall clock ──────────────

    [Fact]
    public void Server_IsFasterThanTheFullReforwardPath()
    {
        var llm = new PredictiveLlm(new DecoderLmConfig(
            260, Dim: 32, NumBlocks: 2, NumHeads: 2, FfDim: 64, MaxSeq: 256, BlockPattern: "TA"));
        var server = llm.CreateServer();

        // Warmup both paths (JIT), then time 120 tokens.
        llm.Generate("ab", 8, new GenerationOptions { StopAtEos = false });
        server.Generate("ab", 8, stopAtEos: false);

        var sw = Stopwatch.StartNew();
        llm.Generate("ab", 120, new GenerationOptions { StopAtEos = false });
        long reforwardMs = sw.ElapsedMilliseconds;

        sw.Restart();
        server.Generate("ab", 120, stopAtEos: false);
        long servedMs = sw.ElapsedMilliseconds;

        Assert.True(servedMs < reforwardMs,
            $"the KV-cache path should win by construction: served {servedMs} ms vs reforward {reforwardMs} ms");
    }
}
