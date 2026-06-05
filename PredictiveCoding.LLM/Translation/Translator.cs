using System.Text.Json;
using PredictiveCoding.LLM.Autograd;
using PredictiveCoding.LLM.Tokenization;
using PredictiveCoding.LLM.Training;

namespace PredictiveCoding.LLM.Translation;

// ─────────────────────────────────────────────────────────────────────────────
// Translator  —  runtime for "our own" decoder-only translator.
//
// Wraps a trained DecoderLM + its byte-level BPE tokenizer and exposes the single
// operation that matters: source text + target-language tag → translated text. It
// also persists/loads itself (manifest + tokenizer + weight checkpoint) so the
// RIASSUNTAZZO library can load a trained model and translate offline.
//
// Decoding is autoregressive greedy/sampling over the DecoderLM forward pass (the
// trainable path has no KV cache, so each step recomputes the prefix — fine for
// the short sentences this model targets).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class Translator
{
    private const string ManifestName  = "manifest.json";
    private const string TokenizerName = "tokenizer.bpe";
    private const string WeightsDir    = "weights";   // FASTER-backed weight checkpoint (a folder)

    private readonly DecoderLM    _model;
    private readonly BpeTokenizer _tok;
    private readonly int          _maxSeq;

    public Translator(DecoderLM model, BpeTokenizer tok)
    {
        _model  = model;
        _tok    = tok;
        _maxSeq = model.Config.MaxSeq;
    }

    public DecoderLmConfig Config => _model.Config;

    /// <summary>
    /// Translates <paramref name="source"/> (any language seen in training) into the
    /// language identified by <paramref name="targetLangTag"/> (e.g. "ita", "eng").
    /// </summary>
    public string Translate(string source, string targetLangTag, int maxNewTokens = 256, float temperature = 0f, Random? rng = null)
    {
        var prompt = TranslationFormat.BuildPrompt(_tok, source, targetLangTag);
        var generated = Decode(prompt, maxNewTokens, temperature, rng);
        return _tok.Decode(generated).Trim();
    }

    /// <summary>Greedy (temperature == 0) or temperature-sampled autoregressive decode; stops at EOS.</summary>
    private int[] Decode(int[] prompt, int maxNew, float temperature, Random? rng)
    {
        var seq = new List<int>(prompt);
        var outp = new List<int>();

        for (int step = 0; step < maxNew && seq.Count < _maxSeq; step++)
        {
            var logits = _model.Forward([.. seq], training: false);   // [seqLen, vocab]
            int last  = seq.Count - 1;
            int vocab = logits.Cols;
            var row   = logits.Data.AsSpan(last * vocab, vocab);

            int next = temperature <= 0f ? Argmax(row) : Sample(row, temperature, rng ??= new Random());
            if (next == _tok.Eos) break;

            seq.Add(next);
            outp.Add(next);
        }
        return [.. outp];
    }

    private static int Argmax(ReadOnlySpan<float> row)
    {
        int best = 0; float bv = float.NegativeInfinity;
        for (int j = 0; j < row.Length; j++) if (row[j] > bv) { bv = row[j]; best = j; }
        return best;
    }

    private static int Sample(ReadOnlySpan<float> logits, float temperature, Random rng)
    {
        int n = logits.Length;
        float max = float.NegativeInfinity;
        for (int j = 0; j < n; j++) max = MathF.Max(max, logits[j]);

        double sum = 0d;
        var probs = new double[n];
        for (int j = 0; j < n; j++) { probs[j] = Math.Exp((logits[j] - max) / temperature); sum += probs[j]; }

        double r = rng.NextDouble() * sum, acc = 0d;
        for (int j = 0; j < n; j++) { acc += probs[j]; if (r <= acc) return j; }
        return n - 1;
    }

    // ── Persistence ──────────────────────────────────────────────────────────────

    private sealed record Manifest(int Vocab, int Dim, int NumBlocks, int NumHeads, int FfDim, int MaxSeq, float Eps);

    /// <summary>Saves manifest + tokenizer + weight checkpoint into <paramref name="dir"/>.</summary>
    public void Save(string dir)
    {
        Directory.CreateDirectory(dir);
        var c = _model.Config;
        var manifest = new Manifest(c.Vocab, c.Dim, c.NumBlocks, c.NumHeads, c.FfDim, c.MaxSeq, c.Eps);
        File.WriteAllText(Path.Combine(dir, ManifestName),
            JsonSerializer.Serialize(manifest, new JsonSerializerOptions { WriteIndented = true }));
        _tok.Save(Path.Combine(dir, TokenizerName));
        Checkpoint.Save(Path.Combine(dir, WeightsDir), _model.Parameters());
    }

    /// <summary>Loads a translator previously written with <see cref="Save"/>.</summary>
    public static Translator Load(string dir)
    {
        var manifest = JsonSerializer.Deserialize<Manifest>(File.ReadAllText(Path.Combine(dir, ManifestName)))
                       ?? throw new InvalidDataException("Empty or invalid translator manifest.");

        var cfg = new DecoderLmConfig(
            manifest.Vocab, manifest.Dim, manifest.NumBlocks, manifest.NumHeads,
            manifest.FfDim, manifest.MaxSeq, Dropout: 0f, Eps: manifest.Eps);

        var tok   = BpeTokenizer.Load(Path.Combine(dir, TokenizerName));
        var model = new DecoderLM(cfg, new Random(0));           // params overwritten by the checkpoint
        Checkpoint.Load(Path.Combine(dir, WeightsDir), model.Parameters());
        return new Translator(model, tok);
    }
}
