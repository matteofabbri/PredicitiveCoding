using System.Diagnostics;
using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Tokenization;
using PredictiveCoding.LLM.Training;

namespace PredictiveCoding.OmniMind.Engine;

/// <summary>Hyper-parameters for the live, from-scratch learning run.</summary>
public sealed record MindOptions(
    int Merges = 700,
    int Dim = 80,
    int Blocks = 2,
    int Heads = 4,
    int FfDim = 192,
    int SeqLen = 48,
    int Steps = 160,
    int Batch = 8,
    float PeakLr = 3e-3f,
    int MaxCorpusChars = 200_000);

/// <summary>
/// Trains the project's own trainable transformer (<see cref="DecoderLM"/>) plus a
/// byte-level BPE tokenizer on the fused corpus, live — then samples a short
/// continuation from the freshly learned weights. Nothing is pre-trained: the
/// "knowledge" comes entirely from what the providers just returned.
/// </summary>
public static class MindTrainer
{
    public sealed record Progress(int Step, int TotalSteps, float Loss, float Perplexity, double StepsPerSec, int Vocab, int Params);

    public sealed record Trained(BpeTokenizer Tokenizer, DecoderLM Model, int Vocab, long Params);

    public static Trained Learn(string corpus, MindOptions opt, Action<Progress> onProgress, Action<string> onLog, CancellationToken ct)
    {
        if (corpus.Length > opt.MaxCorpusChars) corpus = corpus[..opt.MaxCorpusChars];

        onLog($"Training byte-level BPE on {corpus.Length:N0} chars…");
        var tok = new BpeTokenizer();
        tok.Train(corpus, opt.Merges);

        int[] data = tok.Encode(corpus).ToArray();
        onLog($"Vocab {tok.VocabSize} · corpus {data.Length:N0} tokens");

        int seq = Math.Clamp(opt.SeqLen, 8, Math.Max(8, data.Length - 2));
        var cfg = new DecoderLmConfig(tok.VocabSize, opt.Dim, opt.Blocks, opt.Heads, opt.FfDim, Math.Max(seq, 16), Dropout: 0f);
        var model = new DecoderLM(cfg, new Random(0));
        var ps = model.Parameters();
        long paramCount = ps.Sum(t => (long)t.Data.Length);
        onLog($"Model dim={opt.Dim} blocks={opt.Blocks} heads={opt.Heads} · {paramCount:N0} params");

        if (data.Length < seq + 2)
        {
            onLog("Corpus too small to train a sequence model.");
            return new Trained(tok, model, tok.VocabSize, paramCount);
        }

        var adam = new Adam(weightDecay: 0.01f);
        var sched = new LrSchedule
        {
            PeakLr = opt.PeakLr,
            MinLr = opt.PeakLr * 0.1f,
            WarmupSteps = Math.Min(20, opt.Steps / 8 + 1),
            TotalSteps = opt.Steps,
        };
        var rng = new Random(1234);
        var sw = Stopwatch.StartNew();

        for (int step = 1; step <= opt.Steps && !ct.IsCancellationRequested; step++)
        {
            var batch = new List<(int[] tokens, int[] targets)>(opt.Batch);
            for (int b = 0; b < opt.Batch; b++)
            {
                int start = rng.Next(0, data.Length - (seq + 1));
                var t = new int[seq];
                var y = new int[seq];
                Array.Copy(data, start, t, 0, seq);
                Array.Copy(data, start + 1, y, 0, seq);
                batch.Add((t, y));
            }

            float lr = sched.At(step);
            float loss = Trainer.Step(ps, batch, (t, y, train) => model.Loss(t, y, -1, train), adam, lr, clipNorm: 1f);
            float ppl = MathF.Exp(MathF.Min(loss, 15f));
            onProgress(new Progress(step, opt.Steps, loss, ppl,
                step / Math.Max(1e-3, sw.Elapsed.TotalSeconds), tok.VocabSize, (int)Math.Min(int.MaxValue, paramCount)));
        }

        onLog($"Learned in {sw.Elapsed.TotalSeconds:F1}s.");
        return new Trained(tok, model, tok.VocabSize, paramCount);
    }

    /// <summary>Autoregressively samples a continuation; <paramref name="onText"/> receives the
    /// full clean decode after each new token (so the UI streams readable text).</summary>
    public static string Speak(Trained t, string seed, int maxNew, float temperature, int topK, Action<string>? onText, CancellationToken ct)
    {
        DecoderLM model = t.Model;
        BpeTokenizer tok = t.Tokenizer;
        int vocab = model.Config.Vocab;
        int maxSeq = model.Config.MaxSeq;

        var ctx = new List<int>(tok.Encode(seed));
        if (ctx.Count == 0) ctx.Add(32);   // a space byte, to seed generation
        var generated = new List<int>();
        var rng = new Random();

        for (int n = 0; n < maxNew && !ct.IsCancellationRequested; n++)
        {
            int[] window = ctx.Count <= maxSeq ? ctx.ToArray() : ctx.GetRange(ctx.Count - maxSeq, maxSeq).ToArray();
            Tensor logits = model.Forward(window, training: false);

            int last = window.Length - 1;
            var row = new float[vocab];
            Array.Copy(logits.Data, last * vocab, row, 0, vocab);

            int next = SampleRow(row, temperature, topK, rng);
            ctx.Add(next);
            generated.Add(next);
            onText?.Invoke(tok.Decode(generated));
        }

        return tok.Decode(generated);
    }

    private static int SampleRow(float[] logits, float temperature, int topK, Random rng)
    {
        int n = logits.Length;
        float temp = MathF.Max(temperature, 1e-4f);

        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++) if (logits[i] > max) max = logits[i];

        var p = new float[n];
        float sum = 0f;
        for (int i = 0; i < n; i++) { p[i] = MathF.Exp((logits[i] - max) / temp); sum += p[i]; }
        for (int i = 0; i < n; i++) p[i] /= sum;

        if (topK > 0 && topK < n)
        {
            var idx = Enumerable.Range(0, n).ToArray();
            Array.Sort(idx, (a, b) => p[b].CompareTo(p[a]));
            var keep = new bool[n];
            float kept = 0f;
            for (int i = 0; i < topK; i++) { keep[idx[i]] = true; kept += p[idx[i]]; }
            for (int i = 0; i < n; i++) if (!keep[i]) p[i] = 0f;
            if (kept > 0f) for (int i = 0; i < n; i++) p[i] /= kept;
        }

        double r = rng.NextDouble(), acc = 0d;
        for (int i = 0; i < n; i++) { acc += p[i]; if (r <= acc) return i; }
        return n - 1;
    }
}
