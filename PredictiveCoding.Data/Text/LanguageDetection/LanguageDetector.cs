using System.Text;
using System.Text.RegularExpressions;

namespace PredictiveCoding.Data.Text;

// ─────────────────────────────────────────────────────────────────────────────
// LanguageDetector  —  offline language detection for corpus routing/filtering.
//
// Faithful native port of Catalyst's LanguageDetector (Nakatani-Shuyo n-gram naïve
// Bayes): normalise text → extract 1–3 char n-grams → run several randomised
// inference trials over the per-language n-gram probabilities → average. The n-gram
// hash (CaseSensitiveHash32) and the iterative update match Catalyst exactly, so
// results are identical — but with no Catalyst/ML.NET dependency at runtime. The
// model itself is in LanguageProfile (embedded, offline).
//
//   var det = LanguageDetector.CreateDefault();
//   string lang = det.Detect("Ciao, come stai?");          // "Italian"
//   foreach (var c in det.DetectAll(text)) { ... }          // ranked + probabilities
// ─────────────────────────────────────────────────────────────────────────────

public sealed class LanguageDetector
{
    private static readonly Regex NumbersRegex = new(
        @"[\w+\-!?\\/()\[\]{},.:;=$%&]*\d[\w\d+\-!?\\/()\[\]{},.:;=$%&]+", RegexOptions.Compiled);
    private static readonly Regex UrlRegex = new(
        @"https?://[-_.?&~;+=/#0-9A-Za-z]{1,2076}", RegexOptions.Compiled);
    private static readonly Regex EmailRegex = new(
        @"[-_.0-9A-Za-z]{1,64}@[-_0-9A-Za-z]{1,255}[-_.0-9A-Za-z]{1,255}", RegexOptions.Compiled);

    private readonly LanguageProfile _profile;
    private readonly int? _seed;

    public LanguageDetector(LanguageProfile profile, int? seed = null)
    {
        _profile = profile;
        _seed = seed;
    }

    /// <summary>Detector backed by the embedded offline model.</summary>
    /// <param name="seed">Optional RNG seed for reproducible results (detection is randomised).</param>
    public static LanguageDetector CreateDefault(int? seed = null) => new(LanguageProfile.LoadEmbedded(), seed);

    /// <summary>Detector backed by a model file produced by tools/CatalystImport.</summary>
    public static LanguageDetector Load(string path, int? seed = null) => new(LanguageProfile.Load(path), seed);

    // ── Tunables (defaults match Catalyst) ──────────────────────────────────────
    public double Alpha { get; set; } = 0.5;
    public double AlphaWidth { get; set; } = 0.05;
    public int Trials { get; set; } = 7;
    public int NGramLength { get; set; } = 3;
    public int MaxTextLength { get; set; } = 10000;
    public int MaxIterations { get; set; } = 1000;
    public double ProbabilityThreshold { get; set; } = 0.1;
    public double ConvergenceThreshold { get; set; } = 0.99999;
    public int BaseFrequency { get; set; } = 10000;

    /// <summary>Most probable language label, or "Unknown" when nothing scores above the threshold.</summary>
    public string Detect(string text)
    {
        var all = DetectAll(text);
        return all.Count > 0 ? all[0].Language : "Unknown";
    }

    /// <summary>All candidate languages above <see cref="ProbabilityThreshold"/>, highest probability first.</summary>
    public IReadOnlyList<DetectedLanguage> DetectAll(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return Array.Empty<DetectedLanguage>();

        var ngrams = ExtractNGrams(NormalizeText(text));
        if (ngrams.Count == 0) return Array.Empty<DetectedLanguage>();

        int langCount = _profile.Languages.Length;
        var rng = _seed is int s ? new Random(s) : Random.Shared;
        var accumulated = new double[langCount];

        for (int t = 0; t < Trials; t++)
        {
            var prob = new double[langCount];
            Array.Fill(prob, 1.0 / langCount);
            double alpha = Alpha + rng.NextDouble() * AlphaWidth;

            for (int iteration = 0; ; iteration++)
            {
                UpdateProbabilities(prob, ngrams[rng.Next(ngrams.Count)], alpha);
                if (iteration % 5 == 0 &&
                    (NormalizeProbabilities(prob) > ConvergenceThreshold || iteration >= MaxIterations))
                    break;
            }
            for (int j = 0; j < langCount; j++) accumulated[j] += prob[j] / Trials;
        }
        return SortProbabilities(accumulated);
    }

    // prob[i] *= alpha/BaseFrequency + p(ngram | language i)   — sparse, but identical to Catalyst.
    private void UpdateProbabilities(double[] prob, int hash, double alpha)
    {
        if (!_profile.Probabilities.TryGetValue(hash, out var entries)) return;

        double baseProb = alpha / BaseFrequency;
        Span<double> additions = entries.Length <= 64 ? stackalloc double[entries.Length] : new double[entries.Length];
        for (int k = 0; k < entries.Length; k++) additions[k] = prob[entries[k].Lang] * entries[k].Prob;
        for (int i = 0; i < prob.Length; i++) prob[i] *= baseProb;
        for (int k = 0; k < entries.Length; k++) prob[entries[k].Lang] += additions[k];
    }

    private static double NormalizeProbabilities(double[] prob)
    {
        double sum = 0;
        for (int i = 0; i < prob.Length; i++) sum += prob[i];
        double max = 0;
        for (int i = 0; i < prob.Length; i++)
        {
            double v = sum > 0 ? prob[i] / sum : 0;
            if (v > max) max = v;
            prob[i] = v;
        }
        return max;
    }

    private IReadOnlyList<DetectedLanguage> SortProbabilities(double[] prob)
    {
        var result = new List<DetectedLanguage>();
        for (int i = 0; i < prob.Length; i++)
        {
            double p = prob[i];
            if (p <= ProbabilityThreshold) continue;
            int at = result.FindIndex(d => d.Probability < p);
            result.Insert(at < 0 ? result.Count : at, new DetectedLanguage(_profile.Languages[i], p));
        }
        return result;
    }

    // ── n-gram extraction ───────────────────────────────────────────────────────

    private List<int> ExtractNGrams(string text)
    {
        var hashes = new List<int>();
        if (string.IsNullOrEmpty(text)) return hashes;

        var ngram = new NGram();
        foreach (char c in text)
        {
            ngram.Add(c);
            for (int n = 1; n <= NGramLength; n++)
            {
                var span = ngram.Get(n);
                if (span.Length == 0) continue;
                int hash = Hash(span);
                if (_profile.Probabilities.ContainsKey(hash)) hashes.Add(hash);
            }
        }
        return hashes;
    }

    // Catalyst's UID.Hashes.CaseSensitiveHash32 — must match exactly (model keys use it).
    private static int Hash(ReadOnlySpan<char> str)
    {
        int begin = 0, end = str.Length - 1;
        while (begin < str.Length && char.IsWhiteSpace(str[begin])) begin++;
        while (end > 0 && char.IsWhiteSpace(str[end])) end--;

        int a = 352654597, b = a;
        for (int i = begin; i <= end; i += 2)
        {
            a = ((a << 5) + a) ^ str[i];
            if (i == end) break;
            b = ((b << 5) + b) ^ str[i + 1];
        }
        return a + b * 1566083941;
    }

    // ── text normalisation (ported from Catalyst) ───────────────────────────────

    private string NormalizeText(string text)
    {
        text = NormalizeWhitespace(text);
        if (text.Length > MaxTextLength) text = text[..MaxTextLength];
        text = UrlRegex.Replace(text, " ");
        text = EmailRegex.Replace(text, " ");
        text = NumbersRegex.Replace(text, " ");
        text = NormalizeAlphabet(text);
        return NormalizeWhitespace(text);
    }

    // If a text is mostly non-Latin (e.g. CJK/Cyrillic) but carries a few Latin chars,
    // drop the Latin so they don't bias detection toward Latin-script languages.
    private static string NormalizeAlphabet(string text)
    {
        int latin = 0, other = 0;
        foreach (char c in text)
        {
            if (c is <= 'z' and >= 'A') latin++;
            else if (c >= '̀' && (c < 'Ḁ' || c > 'ỿ')) other++;
        }
        if (latin * 2 >= other) return text;

        var sb = new StringBuilder(text.Length);
        foreach (char c in text)
            if (c is > 'z' or < 'A') sb.Append(c);
        return sb.ToString();
    }

    private static string NormalizeWhitespace(string text)
    {
        var sb = new StringBuilder(text.Length);
        bool inWhitespace = false;
        foreach (char c in text)
        {
            if (char.IsWhiteSpace(c))
            {
                if (!inWhitespace) sb.Append(' ');
                inWhitespace = true;
            }
            else
            {
                sb.Append(c);
                inWhitespace = false;
            }
        }
        return sb.ToString();
    }

    // Sliding window of up to 3 chars (space-padded, lower-cased). Skips runs of
    // consecutive capitals (acronyms) by returning empty, exactly like Catalyst.
    private sealed class NGram
    {
        private readonly char[] _buffer = new char[3];
        private int _length;
        private bool _capital;

        public NGram()
        {
            _buffer[0] = ' ';
            _length = 1;
        }

        public void Add(char c)
        {
            char last = _buffer[_length - 1];
            if (last == ' ')
            {
                _buffer[0] = ' ';
                _length = 1;
                _capital = false;
                if (c == ' ') return;
            }
            else if (_length == 3)
            {
                _buffer[0] = _buffer[1];
                _buffer[1] = _buffer[2];
                _length = 2;
            }
            _buffer[_length] = char.ToLowerInvariant(c);
            _length++;
            if (char.IsUpper(c)) { if (char.IsUpper(last)) _capital = true; }
            else _capital = false;
        }

        public ReadOnlySpan<char> Get(int n)
        {
            if (_capital || n < 1 || n > 3 || _length < n) return default;
            if (n == 1)
                return _buffer[_length - 1] == ' ' ? default : Trim(_buffer.AsSpan(_length - 1, 1));
            return Trim(_buffer.AsSpan(_length - n, n));
        }

        private static ReadOnlySpan<char> Trim(ReadOnlySpan<char> span)
        {
            int start = 0, end = span.Length - 1;
            while (start <= end && char.IsWhiteSpace(span[start])) start++;
            while (end >= start && char.IsWhiteSpace(span[end])) end--;
            return start > end ? default : span.Slice(start, end - start + 1);
        }
    }
}
