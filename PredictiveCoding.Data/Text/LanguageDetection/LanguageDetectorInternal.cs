// Copyright 2014 Pēteris Ņikiforovs
// Copyright 2014 Nakatani Shuyo
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace PredictiveCoding.Data.Text;

// ─────────────────────────────────────────────────────────────────────────────
// LanguageDetectorInternal  —  the langdetect core (Nakatani-Shuyo n-gram naïve
// Bayes with randomised inference trials), C# port by Pēteris Ņikiforovs, imported
// from Maddalena.Cognitive where it ran in production. Works on the 53 embedded
// n-gram profiles and returns ISO codes; LanguageDetector wraps it with names.
//
// Faithful copy — the algorithm is untouched. Deliberate adaptations:
//   · Newtonsoft.Json → System.Text.Json (no external dependency);
//   · the profile set and the n-gram → per-language probability map are built once
//     per process (they were per-instance) — NGramLength was effectively the
//     constant 3, so the map does not depend on instance settings;
//   · DetectAll returns (code, probability) tuples instead of a nested class.
// ─────────────────────────────────────────────────────────────────────────────

internal sealed class LanguageDetectorInternal
{
    /// <summary>n-gram order cap — profiles only contain 1–3 char n-grams.</summary>
    private const int NGramLength = 3;

    private static readonly Regex UrlRegex = new(
        "https?://[-_.?&~;+=/#0-9A-Za-z]{1,2076}", RegexOptions.Compiled);
    private static readonly Regex EmailRegex = new(
        "[-_.0-9A-Za-z]{1,64}@[-_0-9A-Za-z]{1,255}[-_.0-9A-Za-z]{1,255}", RegexOptions.Compiled);

    private static readonly Lazy<Model> Loaded = new(LoadModel);

    /// <summary>The embedded profiles, resolved: codes plus n-gram → (language index → P(ngram | language)).</summary>
    private sealed record Model(string[] Codes, Dictionary<string, Dictionary<int, double>> WordLanguageProbabilities);

    // ── Tunables (defaults are the original langdetect values) ──────────────────
    public double Alpha { get; set; } = 0.5;
    public int? RandomSeed { get; set; }
    public int Trials { get; set; } = 7;
    public int MaxTextLength { get; set; } = 10000;
    public double AlphaWidth { get; set; } = 0.05;
    public int MaxIterations { get; set; } = 1000;
    public double ProbabilityThreshold { get; set; } = 0.1;
    public double ConvergenceThreshold { get; set; } = 0.99999;
    public int BaseFrequency { get; set; } = 10000;

    /// <summary>All ISO codes the embedded profiles cover, sorted ordinally.</summary>
    public static IReadOnlyList<string> ProfileCodes => Loaded.Value.Codes;

    private static Model LoadModel()
    {
        var assembly = typeof(LanguageDetectorInternal).Assembly;
        var profiles = assembly.GetManifestResourceNames()
            .Where(name => name.Contains(".Resources.LangProfiles.", StringComparison.Ordinal))
            .Select(name =>
            {
                using var stream = assembly.GetManifestResourceStream(name)!;
                return JsonSerializer.Deserialize<LanguageProfile>(stream)
                       ?? throw new InvalidDataException($"Empty language profile resource '{name}'.");
            })
            .OrderBy(p => p.Code, StringComparer.Ordinal)
            .ToList();
        if (profiles.Count == 0)
            throw new InvalidOperationException(
                "No embedded language profiles found under Resources/LangProfiles — check the .csproj EmbeddedResource entries.");

        var map = new Dictionary<string, Dictionary<int, double>>();
        for (int i = 0; i < profiles.Count; i++)
        {
            LanguageProfile profile = profiles[i];
            foreach (var (word, count) in profile.Frequencies)
            {
                if (word.Length < 1 || word.Length > NGramLength) continue;
                if (!map.TryGetValue(word, out var perLanguage))
                    map[word] = perLanguage = new Dictionary<int, double>();
                perLanguage[i] = (double)count / profile.WordCount[word.Length - 1];
            }
        }
        return new Model(profiles.Select(p => p.Code).ToArray(), map);
    }

    /// <summary>Most probable ISO code, or null when nothing scores above <see cref="ProbabilityThreshold"/>.</summary>
    public string? Detect(string text)
    {
        var all = DetectAll(text);
        return all.Count > 0 ? all[0].Code : null;
    }

    /// <summary>All candidate codes above <see cref="ProbabilityThreshold"/>, highest probability first.</summary>
    public IReadOnlyList<(string Code, double Probability)> DetectAll(string text)
    {
        Model model = Loaded.Value;
        var ngrams = ExtractNGrams(NormalizeText(text), model.WordLanguageProbabilities);
        if (ngrams.Count == 0) return Array.Empty<(string, double)>();

        int languageCount = model.Codes.Length;
        var accumulated = new double[languageCount];
        var random = RandomSeed is int seed ? new Random(seed) : new Random();

        for (int t = 0; t < Trials; t++)
        {
            var probs = new double[languageCount];
            Array.Fill(probs, 1.0 / languageCount);
            double alpha = Alpha + random.NextDouble() * AlphaWidth;

            for (int i = 0; ; i++)
            {
                UpdateProbabilities(probs, ngrams[random.Next(ngrams.Count)], alpha, model);
                if (i % 5 == 0 && (NormalizeProbabilities(probs) > ConvergenceThreshold || i >= MaxIterations))
                    break;
            }

            for (int j = 0; j < languageCount; j++)
                accumulated[j] += probs[j] / Trials;
        }

        return SortProbabilities(accumulated, model.Codes);
    }

    private static List<string> ExtractNGrams(
        string text, Dictionary<string, Dictionary<int, double>> known)
    {
        var list = new List<string>();
        var ngram = new NGram();
        foreach (char c in text)
        {
            ngram.Add(c);
            for (int n = 1; n <= NGramLength; n++)
            {
                string? w = ngram.Get(n);
                if (w is not null && known.ContainsKey(w)) list.Add(w);
            }
        }
        return list;
    }

    // ── Text normalisation ──────────────────────────────────────────────────────

    private string NormalizeText(string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return "";
        if (text.Length > MaxTextLength) text = text[..MaxTextLength];

        text = UrlRegex.Replace(text, " ");
        text = EmailRegex.Replace(text, " ");
        text = NormalizeAlphabet(text);
        text = NormalizeWhitespace(text);
        return text;
    }

    // If a text is mostly non-Latin (e.g. CJK/Cyrillic) but carries a few Latin chars,
    // drop the Latin so they don't bias detection toward Latin-script languages.
    private static string NormalizeAlphabet(string text)
    {
        int latinCount = 0, nonLatinCount = 0;
        foreach (char c in text)
        {
            if (c is <= 'z' and >= 'A') latinCount++;
            else if (c >= '̀' && !(c >= 0x1e00 && c <= 0x1eff)) nonLatinCount++;
        }

        if (latinCount * 2 >= nonLatinCount) return text;

        var sb = new StringBuilder(text.Length);
        foreach (char c in text)
            if (c is > 'z' or < 'A') sb.Append(c);
        return sb.ToString();
    }

    private static string NormalizeWhitespace(string text)
    {
        var sb = new StringBuilder(text.Length);
        char? prev = null;
        foreach (char c in text)
        {
            if (c != ' ' || prev != ' ') sb.Append(c);
            prev = c;
        }
        return sb.ToString();
    }

    // ── Probabilities ───────────────────────────────────────────────────────────

    // prob[i] *= alpha/BaseFrequency + P(ngram | language i)
    private void UpdateProbabilities(double[] prob, string word, double alpha, Model model)
    {
        if (!model.WordLanguageProbabilities.TryGetValue(word, out var perLanguage)) return;

        double weight = alpha / BaseFrequency;
        for (int i = 0; i < prob.Length; i++)
            prob[i] *= weight + (perLanguage.TryGetValue(i, out double p) ? p : 0);
    }

    private static double NormalizeProbabilities(double[] probs)
    {
        double sum = 0;
        foreach (double p in probs) sum += p;

        double max = 0;
        for (int i = 0; i < probs.Length; i++)
        {
            double p = probs[i] / sum;
            if (p > max) max = p;
            probs[i] = p;
        }
        return max;
    }

    private IReadOnlyList<(string Code, double Probability)> SortProbabilities(double[] probs, string[] codes)
    {
        var list = new List<(string Code, double Probability)>();
        for (int j = 0; j < probs.Length; j++)
        {
            double p = probs[j];
            if (p <= ProbabilityThreshold) continue;
            int at = list.FindIndex(x => x.Probability < p);
            list.Insert(at < 0 ? list.Count : at, (codes[j], p));
        }
        return list;
    }

    // Sliding window of up to 3 chars. Word boundaries stay as spaces (profiles key
    // n-grams like " ca" / "ne "); runs of consecutive capitals (acronyms) yield null.
    private sealed class NGram
    {
        private StringBuilder _buffer = new(" ", NGramLength);
        private bool _capital;

        public void Add(char c)
        {
            char last = _buffer[^1];
            if (last == ' ')
            {
                _buffer = new StringBuilder(" ");
                _capital = false;
                if (c == ' ') return;
            }
            else if (_buffer.Length >= NGramLength)
            {
                _buffer.Remove(0, 1);
            }

            _buffer.Append(c);

            if (char.IsUpper(c)) { if (char.IsUpper(last)) _capital = true; }
            else _capital = false;
        }

        public string? Get(int n)
        {
            if (_capital || n < 1 || n > NGramLength || _buffer.Length < n) return null;
            if (n == 1)
            {
                char c = _buffer[^1];
                return c == ' ' ? null : c.ToString();
            }
            return _buffer.ToString(_buffer.Length - n, n);
        }
    }
}
