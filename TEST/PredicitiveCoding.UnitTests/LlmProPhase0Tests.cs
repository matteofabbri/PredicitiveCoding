using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Eval;
using PredictiveCoding.LLM.Tokenization;

namespace PredicitiveCoding.UnitTests;

// Pro roadmap Phase 0: a real tokenizer — trained BPE that travels with the
// model, fingerprint-verified, measured in bits per byte (the only metric that
// can compare tokenizers).
public class LlmProPhase0Tests : IDisposable
{
    private readonly string _tempRoot = Path.Combine(Path.GetTempPath(), $"pcbpe_{Guid.NewGuid():N}");

    public void Dispose()
    {
        if (Directory.Exists(_tempRoot)) Directory.Delete(_tempRoot, recursive: true);
    }

    // ── The tokenizer itself ─────────────────────────────────────────────────

    [Fact]
    public void Bpe_Training_IsLosslessCompressiveAndDeterministic()
    {
        string corpus = EvalCorpus.Tiny;

        var a = new BpeTokenizer(); a.Train(corpus, numMerges: 300);
        var b = new BpeTokenizer(); b.Train(corpus, numMerges: 300);

        // Deterministic: two trainings on the same corpus agree token for token.
        Assert.Equal(a.Encode(corpus), b.Encode(corpus));
        Assert.Equal(a.VocabSize, b.VocabSize);

        // Lossless round-trip, Unicode included (byte-level BPE eats anything).
        foreach (string text in (string[])[corpus, "ciao, è già un'ipotesi 🧠 →", "The brain is a prediction machine."])
            Assert.Equal(text, a.Decode(a.Encode(text)));

        // Compressive: fewer tokens than bytes on the corpus it learned.
        int bytes = System.Text.Encoding.UTF8.GetByteCount(corpus);
        Assert.True(a.Encode(corpus).Count < bytes / 2,
            $"BPE barely compressed: {a.Encode(corpus).Count} tokens for {bytes} bytes");
    }

    // ── The tokenizer travels with the model ─────────────────────────────────

    [Fact]
    public void SaveLoad_BpeModel_ReconstructsTheExactTokenizer()
    {
        var llm = PredictiveLlm.Create2026(EvalCorpus.Tiny, merges: 200,
            dim: 16, blocks: 2, heads: 2, ffDim: 32, maxSeq: 48);
        Assert.True(llm.Config.Vocab > 260, "BPE preset should grow the vocab past the byte base");

        string dir = Path.Combine(_tempRoot, "bpe-model");
        llm.Save(dir);
        // Tutto lo stato vive nel FASTER store unificato (niente piu' file separati).

        var back = PredictiveLlm.Load(dir);                        // no tokenizer passed: it ships with the model
        Assert.Equal(llm.Config.Vocab, back.Tokenizer.VocabSize);
        Assert.Equal(
            llm.Tokenizer.Encode("prediction machine"),
            back.Tokenizer.Encode("prediction machine"));          // identical merges
        Assert.Equal(llm.Generate("The brain", 6), back.Generate("The brain", 6));

        // Passing a tokenizer to a BPE model is refused loudly (ambiguity kills).
        Assert.Throws<ArgumentException>(() => PredictiveLlm.Load(dir, new ByteTokenizer()));
    }

    [Fact]
    public void Load_BrokenCheckpoint_FailsLoudly()
    {
        var llm = PredictiveLlm.Create2026(EvalCorpus.Tiny, merges: 100,
            dim: 16, blocks: 1, heads: 2, ffDim: 32, maxSeq: 32);
        string dir = Path.Combine(_tempRoot, "tampered");
        llm.Save(dir);

        // Tutto lo stato vive nel FASTER store unificato (vocab compreso, con
        // fingerprint verificato al Load). Un checkpoint monco — qui manca l'hint
        // di page-size, come dopo un salvataggio interrotto — dev'essere rifiutato
        // rumorosamente, non caricato a pezzi.
        File.Delete(Path.Combine(dir, "pagesize"));
        Assert.ThrowsAny<Exception>(() => PredictiveLlm.Load(dir));
    }

    [Fact]
    public void SaveLoad_ByteModel_StillNeedsNothing()
    {
        var llm = new PredictiveLlm(new PredictiveCoding.LLM.Autograd.DecoderLmConfig(260, 16, 1, 2, 32, 32));
        string dir = Path.Combine(_tempRoot, "byte-model");
        llm.Save(dir);

        Assert.False(File.Exists(Path.Combine(dir, "tokenizer.bpe")));   // stateless: no vocab file
        var back = PredictiveLlm.Load(dir);
        Assert.Equal(260, back.Tokenizer.VocabSize);

        // Legacy directory (saved before Phase 0): no manifest → byte fallback.
        File.Delete(Path.Combine(dir, "tokenizer.json"));
        Assert.Equal(260, PredictiveLlm.Load(dir).Tokenizer.VocabSize);
    }

    // ── The metric: bits per byte, tokenizer-independent ─────────────────────

    [Fact]
    public void BitsPerByte_BpeBeatsBytes_AfterEqualTraining()
    {
        string corpus = EvalCorpus.Tiny;
        var opts = new TrainingOptions { SeqLen = 24, Stride = 12, Epochs = 5, BatchWindows = 6 };

        var byteLlm = new PredictiveLlm(new PredictiveCoding.LLM.Autograd.DecoderLmConfig(
            260, Dim: 16, NumBlocks: 2, NumHeads: 2, FfDim: 32, MaxSeq: 48, BlockPattern: "TA",
            NumLoops: 2, MtpWeight: 0.3f, NumExperts: 2, MoeTopK: 1, UsePlan: true), seed: 42);
        byteLlm.Train(corpus, opts);
        var byteScore = byteLlm.EvaluateBitsPerByte(corpus);

        var bpeLlm = PredictiveLlm.Create2026(corpus, merges: 300,
            dim: 16, blocks: 2, heads: 2, ffDim: 32, maxSeq: 48);
        bpeLlm.Train(corpus, opts);
        var bpeScore = bpeLlm.EvaluateBitsPerByte(corpus);

        Assert.True(double.IsFinite(byteScore.BitsPerByte) && double.IsFinite(bpeScore.BitsPerByte));
        Assert.True(bpeScore.Tokens < byteScore.Tokens,            // same text, far fewer predictions
            $"BPE did not shorten the sequence: {bpeScore.Tokens} vs {byteScore.Tokens} tokens");
        Assert.True(bpeScore.BitsPerByte < byteScore.BitsPerByte,
            $"BPE should win bits/byte on its own corpus: {bpeScore} vs {byteScore}");
    }
}
