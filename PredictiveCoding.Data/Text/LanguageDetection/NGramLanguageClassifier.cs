using System.IO.Compression;
using System.Text;

namespace PredictiveCoding.Data.Text;

// ─────────────────────────────────────────────────────────────────────────────
// NGramLanguageClassifier  —  trainable, native character-n-gram naïve-Bayes
// language identifier. Pure BCL, no external dependencies.
//
// Score(lang | text) = logPrior(lang) + Σ_g count_text(g) · logP(g | lang),
// with Lidstone smoothing and a per-language fallback for unseen n-grams. The
// best-scoring language wins; PredictAll returns a softmax probability ranking.
//
// Unlike the fixed langdetect profiles behind LanguageDetector, this is trained on
// whatever corpus you give it, so it can cover languages and domains the embedded
// profiles don't — and its accuracy scales with the data.
//
//   var clf  = NGramLanguageClassifier.CreateDefault();   // trained on the bundled seed corpus
//   string l = clf.Predict("Hola, ¿qué tal?");            // "Spanish"
// ─────────────────────────────────────────────────────────────────────────────

public sealed class NGramLanguageClassifier
{
    private const uint Magic = 0x4E474C43;   // "NGLC"

    private readonly int _minN;
    private readonly int _maxN;
    private readonly string[] _languages;
    private readonly double[] _logPrior;
    private readonly Dictionary<string, float>[] _logProb;
    private readonly float[] _fallback;

    internal NGramLanguageClassifier(
        int minN, int maxN, string[] languages,
        double[] logPrior, Dictionary<string, float>[] logProb, float[] fallback)
    {
        _minN = minN;
        _maxN = maxN;
        _languages = languages;
        _logPrior = logPrior;
        _logProb = logProb;
        _fallback = fallback;
    }

    /// <summary>The languages this model can recognise.</summary>
    public IReadOnlyList<string> Languages => _languages;

    /// <summary>Most probable language label, or "Unknown" for empty/letter-less input.</summary>
    public string Predict(string text)
    {
        var ranked = PredictAll(text);
        return ranked.Count > 0 ? ranked[0].Language : "Unknown";
    }

    /// <summary>All languages ranked by probability (softmax over naïve-Bayes log-scores), highest first.</summary>
    public IReadOnlyList<DetectedLanguage> PredictAll(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return Array.Empty<DetectedLanguage>();
        var grams = NGrams.Extract(text, _minN, _maxN);
        if (grams.Count == 0) return Array.Empty<DetectedLanguage>();

        int n = _languages.Length;
        var score = new double[n];
        for (int i = 0; i < n; i++)
        {
            double s = _logPrior[i];
            var lp = _logProb[i];
            float fb = _fallback[i];
            foreach (var (gram, count) in grams)
                s += count * (lp.TryGetValue(gram, out float v) ? v : fb);
            score[i] = s;
        }

        double max = double.NegativeInfinity;
        for (int i = 0; i < n; i++) if (score[i] > max) max = score[i];
        double sum = 0;
        var prob = new double[n];
        for (int i = 0; i < n; i++) { prob[i] = Math.Exp(score[i] - max); sum += prob[i]; }

        var result = new List<DetectedLanguage>(n);
        for (int i = 0; i < n; i++) result.Add(new DetectedLanguage(_languages[i], prob[i] / sum));
        result.Sort((a, b) => b.Probability.CompareTo(a.Probability));
        return result;
    }

    // ── Default model (trained from the bundled seed corpus) ─────────────────────

    /// <summary>
    /// Builds a classifier trained on the embedded seed corpus (Resources/language-samples.tsv).
    /// Good for the well-resourced languages it ships with; for more/better coverage, train your
    /// own with <see cref="NGramLanguageClassifierTrainer"/> and <see cref="Save"/>/<see cref="Load"/>.
    /// </summary>
    public static NGramLanguageClassifier CreateDefault()
    {
        var trainer = new NGramLanguageClassifierTrainer();
        foreach (var (language, text) in LanguageSamples.Default())
            trainer.Add(language, text);
        return trainer.Train();
    }

    // ── Persistence (for models trained on large corpora) ────────────────────────

    /// <summary>Saves the trained model (deflate-compressed) to <paramref name="path"/>.</summary>
    public void Save(string path)
    {
        using var fs = File.Create(path);
        using var deflate = new DeflateStream(fs, CompressionLevel.SmallestSize);
        using var bw = new BinaryWriter(deflate, Encoding.UTF8);

        bw.Write(Magic);
        bw.Write(1);                       // format version
        bw.Write(_minN);
        bw.Write(_maxN);
        bw.Write(_languages.Length);
        foreach (var lang in _languages) bw.Write(lang);
        for (int i = 0; i < _languages.Length; i++)
        {
            bw.Write(_logPrior[i]);
            bw.Write(_fallback[i]);
            bw.Write(_logProb[i].Count);
            foreach (var (gram, lp) in _logProb[i]) { bw.Write(gram); bw.Write(lp); }
        }
    }

    /// <summary>Loads a model saved with <see cref="Save"/>.</summary>
    public static NGramLanguageClassifier Load(string path)
    {
        using var fs = File.OpenRead(path);
        using var deflate = new DeflateStream(fs, CompressionMode.Decompress);
        using var br = new BinaryReader(deflate, Encoding.UTF8);

        if (br.ReadUInt32() != Magic) throw new InvalidDataException("Not an NGram language model.");
        if (br.ReadInt32() != 1) throw new InvalidDataException("Unsupported model format.");
        int minN = br.ReadInt32();
        int maxN = br.ReadInt32();
        int n = br.ReadInt32();

        var languages = new string[n];
        for (int i = 0; i < n; i++) languages[i] = br.ReadString();

        var logPrior = new double[n];
        var fallback = new float[n];
        var logProb = new Dictionary<string, float>[n];
        for (int i = 0; i < n; i++)
        {
            logPrior[i] = br.ReadDouble();
            fallback[i] = br.ReadSingle();
            int entries = br.ReadInt32();
            var lp = new Dictionary<string, float>(entries, StringComparer.Ordinal);
            for (int e = 0; e < entries; e++)
            {
                string gram = br.ReadString();
                lp[gram] = br.ReadSingle();
            }
            logProb[i] = lp;
        }
        return new NGramLanguageClassifier(minN, maxN, languages, logPrior, logProb, fallback);
    }
}
