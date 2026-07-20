using PredictiveCoding.LLM;
using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Data;
using PredictiveCoding.LLM.Tokenization;

namespace PredicitiveCoding.UnitTests;

// ROADMAP-Scale Phase B: scale the model. B1 — real config presets (the
// embedding becomes a row gather, Small/Base presets size the model for one
// RTX 4070). B2 — a real BPE tokenizer: word-split training over unique
// pieces, scalable to 16k-32k vocabularies, v1 files still bit-identical.
public class LlmScaleBTests
{
    // ── Embedding: gather ≡ one-hot matmul ───────────────────────────────────

    [Fact]
    public void EmbeddingGather_MatchesOneHotMatMul_ForwardAndGradient()
    {
        const int vocab = 11, dim = 5;
        int[] tokens = [3, 7, 3, 0, 10, 3];   // repeats on purpose: the scatter must accumulate

        var rng = new Random(123);
        var wembA = new Tensor(vocab, dim);
        for (int i = 0; i < wembA.Data.Length; i++) wembA.Data[i] = (float)(rng.NextDouble() * 2 - 1);
        var wembB = new Tensor(vocab, dim);
        Array.Copy(wembA.Data, wembB.Data, wembA.Data.Length);

        // Path A — the old formulation: onehot(tokens)·Wemb.
        var oneHot = new Tensor(tokens.Length, vocab);
        for (int i = 0; i < tokens.Length; i++) oneHot.Data[i * vocab + tokens[i]] = 1f;
        var outA = Ag.MatMul(oneHot, wembA);

        // Path B — the new formulation: gather of the token rows.
        var outB = Ag.GatherRows(wembB, tokens);

        Assert.Equal(outA.Data, outB.Data);   // forward bit-identical

        Ag.Mean(outA).Backward();
        Ag.Mean(outB).Backward();
        for (int i = 0; i < wembA.Grad.Length; i++)
            Assert.Equal(wembA.Grad[i], wembB.Grad[i], precision: 7);
    }

    // ── Presets: exact parameter counts ──────────────────────────────────────

    [Fact]
    public void SmallPreset_ParameterCount_MatchesTheFormula()
    {
        var cfg   = DecoderLmConfig.Small(vocab: 16384);
        var model = new DecoderLM(cfg, new Random(1));

        Assert.Equal(cfg.DenseParameterFormula(), model.ParameterCount);
        Assert.Equal(26_767_104L, model.ParameterCount);          // ~30M-class @ 16k vocab
        Assert.Equal(64, cfg.HeadDim);                            // the standard head width
        Assert.Equal(512, cfg.MaxSeq);
    }

    [Fact]
    public void FormulaAgreesWithInstantiation_OnAnArbitraryDenseConfig()
    {
        // Cross-check the closed-form count against a real instance on a config
        // with nothing round about it — guards the formula, cheaply.
        var cfg   = new DecoderLmConfig(Vocab: 301, Dim: 24, NumBlocks: 3, NumHeads: 4, FfDim: 56, MaxSeq: 32);
        var model = new DecoderLM(cfg, new Random(2));

        Assert.Equal(cfg.DenseParameterFormula(), model.ParameterCount);
    }

    [Fact]
    public void BasePreset_FormulaGivesTheGpt2SmallClassCount()
    {
        // Not instantiated (135M params ≈ 1 GB with grads); the formula is
        // verified against real instances by the two tests above.
        var cfg = DecoderLmConfig.Base(vocab: 32768);

        Assert.Equal(135_350_784L, cfg.DenseParameterFormula());  // GPT-2-small class
        Assert.Equal(64, cfg.HeadDim);
        Assert.Equal(1024, cfg.MaxSeq);
    }

    [Fact]
    public void DenseParameterFormula_RefusesNonDenseConfigs()
    {
        var moe = DecoderLmConfig.Small(1000) with { NumExperts = 4 };
        Assert.Throws<InvalidOperationException>(() => moe.DenseParameterFormula());
    }

    // ── The factories wire the tokenizer in ──────────────────────────────────

    [Fact]
    public void CreateSmall_TakesVocabFromTokenizer_AndRuns()
    {
        var llm = PredictiveLlm.CreateSmall(new ByteTokenizer(), maxSeq: 64);

        Assert.Equal(260, llm.Config.Vocab);
        Assert.Equal(384, llm.Config.Dim);
        Assert.Equal(0, llm.Config.NumExperts);   // dense: the showcase features stay off
        Assert.Equal(1, llm.Config.NumLoops);

        var logits = llm.Model.Forward([1, 2, 3]);
        Assert.Equal(3, logits.Rows);
        Assert.Equal(260, logits.Cols);
        Assert.All(logits.Data, v => Assert.True(float.IsFinite(v)));
    }

    // ═════ B2 — the real BPE tokenizer (word-split, scalable) ════════════════

    private static string SyntheticCorpus(int lines = 400)
    {
        // Repetitive, structured text: BPE has something real to learn.
        string[] sentences =
        [
            "the brain is a prediction machine that minimizes surprise.",
            "predictive coding explains perception as inference over causes.",
            "the cortex learns a model of the world from prediction errors.",
            "ogni area corticale predice il proprio ingresso, è già un'ipotesi.",
            "numbers like 12345 and symbols #@! survive byte-level tokenization.",
        ];
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < lines; i++) sb.AppendLine(sentences[i % sentences.Length]);
        return sb.ToString();
    }

    [Fact]
    public void TrainFromCorpus_LearnsAVocab_LosslessAndCompressive()
    {
        string corpus = SyntheticCorpus();
        var tok = new BpeTokenizer();
        tok.TrainFromCorpus(CorpusSource.FromText(corpus), targetVocab: 600);

        Assert.True(tok.WordSplit);
        Assert.True(tok.VocabSize > 300, $"vocab barely grew: {tok.VocabSize}");

        // Lossless round-trip, including text the tokenizer never saw.
        foreach (string text in (string[])[
            "the brain is a prediction machine.",
            "ciao, è già un'ipotesi 🧠 → sorpresa!\n\nnuova riga\te tab",
            "  leading spaces, trailing  ", ""])
            Assert.Equal(text, tok.Decode(tok.Encode(text)));

        // Compressive on in-distribution text: clearly fewer tokens than bytes.
        string sample = "the cortex learns a model of the world from prediction errors.";
        int bytes = System.Text.Encoding.UTF8.GetByteCount(sample);
        Assert.True(tok.Encode(sample).Count < bytes / 2,
            $"{tok.Encode(sample).Count} tokens for {bytes} bytes");
    }

    [Fact]
    public void TrainFromCorpus_MergesStayInsideWordPieces()
    {
        var tok = new BpeTokenizer();
        tok.TrainFromCorpus(CorpusSource.FromText(SyntheticCorpus()), targetVocab: 600);

        // Every learned token is either all-whitespace or has at most a single
        // leading space and no internal whitespace — the GPT-2 piece contract.
        bool sawSpaceWord = false;
        for (int id = 260; id < tok.VocabSize; id++)
        {
            string s = tok.Decode([id], skipSpecial: false);
            if (s.All(char.IsWhiteSpace)) continue;
            string body = s[0] == ' ' ? s[1..] : s;
            Assert.DoesNotContain(body, c => char.IsWhiteSpace(c));
            if (s[0] == ' ' && s.Length > 1) sawSpaceWord = true;
        }
        // The " word" convention actually materialized (e.g. " the").
        Assert.True(sawSpaceWord, "no space-prefixed word token was ever learned");
    }

    [Fact]
    public void TrainFromCorpus_IsDeterministic()
    {
        string corpus = SyntheticCorpus();
        var a = new BpeTokenizer(); a.TrainFromCorpus(CorpusSource.FromText(corpus), 500);
        var b = new BpeTokenizer(); b.TrainFromCorpus(CorpusSource.FromText(corpus), 500);

        Assert.Equal(a.VocabSize, b.VocabSize);
        Assert.Equal(a.Encode(corpus), b.Encode(corpus));
    }

    [Fact]
    public void TrainFromCorpus_SaveLoad_PreservesModeAndEncoding()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"pcbpe2_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            string corpus = SyntheticCorpus();
            var tok = new BpeTokenizer();
            tok.TrainFromCorpus(CorpusSource.FromText(corpus), 500);

            string path = Path.Combine(dir, "tok.bpe");
            tok.Save(path);
            var back = BpeTokenizer.Load(path);

            Assert.True(back.WordSplit);
            Assert.Equal(tok.VocabSize, back.VocabSize);
            string probe = "the brain predicts, è già un'ipotesi 🧠";
            Assert.Equal(tok.Encode(probe), back.Encode(probe));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void WholeStreamV1_SaveLoad_StaysBitIdentical()
    {
        string dir = Path.Combine(Path.GetTempPath(), $"pcbpe1_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);
        try
        {
            string corpus = SyntheticCorpus(80);
            var tok = new BpeTokenizer();
            tok.Train(corpus, numMerges: 200);   // legacy whole-stream mode

            string path = Path.Combine(dir, "tok.bpe");
            tok.Save(path);
            var back = BpeTokenizer.Load(path);

            Assert.False(back.WordSplit);        // v1 magic → whole-stream encoder
            string probe = "the brain is a prediction machine.";
            Assert.Equal(tok.Encode(probe), back.Encode(probe));
        }
        finally { Directory.Delete(dir, recursive: true); }
    }

    [Fact]
    public void TrainingModes_RefuseToMix()
    {
        var trained = new BpeTokenizer();
        trained.Train("aaabbbaaabbb", 10);
        Assert.Throws<InvalidOperationException>(
            () => trained.TrainFromCorpus(CorpusSource.FromText("x"), 300));

        var split = new BpeTokenizer();
        split.TrainFromCorpus(CorpusSource.FromText(SyntheticCorpus(40)), 300);
        Assert.Throws<InvalidOperationException>(() => split.Train("aaabbb", 10));
    }

    [Fact]
    public void TrainFromCorpus_HandlesThousandsOfMerges()
    {
        // ~360KB corpus, 2000 merges: the incremental trainer does this in
        // seconds — the old whole-stream pass-per-merge would take minutes
        // (if this test ever hangs, the scaling regressed).
        var rng = new Random(7);
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < 4000; i++)
        {
            for (int w = 0; w < 12; w++) sb.Append(" word").Append(rng.Next(3000));
            sb.AppendLine(".");
        }
        var tok = new BpeTokenizer();
        int reported = 0;
        tok.TrainFromCorpus(CorpusSource.FromText(sb.ToString()), targetVocab: 2260,
                            onProgress: (done, target) => reported = done);

        Assert.Equal(2260, tok.VocabSize);       // all 2000 merges found (corpus is repetitive)
        Assert.Equal(2000, reported);
        Assert.Equal(" word219", tok.Decode(tok.Encode(" word219")));
    }

    // ═════ B3 — the memory budget, measured ══════════════════════════════════

    [Fact]
    public void MemoryBudget_ParamsMatchTheFormula_AndActivationsAreReal()
    {
        var cfg = new DecoderLmConfig(Vocab: 300, Dim: 32, NumBlocks: 2, NumHeads: 4, FfDim: 64, MaxSeq: 64);
        var report = PredictiveCoding.LLM.Training.MemoryBudget.Measure(cfg, seqLen: 48);

        Assert.Equal(cfg.DenseParameterFormula() * 4, report.ParameterBytes);
        Assert.Equal(report.ParameterBytes, report.GradientBytes);
        Assert.Equal(2 * report.ParameterBytes, report.AdamStateBytes);
        Assert.True(report.ActivationBytes > 0);
        Assert.Equal(report.ParameterBytes + report.GradientBytes + report.AdamStateBytes + report.ActivationBytes,
                     report.TotalBytes);
        Assert.True(report.ProjectFp16Bytes < report.TotalBytes);   // mixed precision must actually save
    }

    [Fact]
    public void MemoryBudget_ActivationsGrowWithSequence_AndMeasureIsDeterministic()
    {
        var cfg = new DecoderLmConfig(Vocab: 300, Dim: 32, NumBlocks: 2, NumHeads: 4, FfDim: 64, MaxSeq: 64);

        var s32 = PredictiveCoding.LLM.Training.MemoryBudget.Measure(cfg, seqLen: 32);
        var s64 = PredictiveCoding.LLM.Training.MemoryBudget.Measure(cfg, seqLen: 64);
        Assert.True(s64.ActivationBytes > s32.ActivationBytes);

        var again = PredictiveCoding.LLM.Training.MemoryBudget.Measure(cfg, seqLen: 32);
        Assert.Equal(s32.ActivationBytes, again.ActivationBytes);
    }

    [Fact]
    public void Encode_WordSplit_IsThreadSafe()
    {
        var tok = new BpeTokenizer();
        tok.TrainFromCorpus(CorpusSource.FromText(SyntheticCorpus()), 500);

        string[] texts = [.. Enumerable.Range(0, 200)
            .Select(i => $"the brain {i} predicts prediction errors, è già un'ipotesi {i % 7}.")];
        var expected = texts.Select(t => tok.Encode(t).ToArray()).ToArray();

        var got = new int[texts.Length][];
        Parallel.For(0, texts.Length, i => got[i] = [.. tok.Encode(texts[i])]);

        for (int i = 0; i < texts.Length; i++)
            Assert.Equal(expected[i], got[i]);
    }
}
