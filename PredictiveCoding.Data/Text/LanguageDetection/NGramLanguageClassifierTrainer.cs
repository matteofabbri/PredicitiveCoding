namespace PredictiveCoding.Data.Text;

// ─────────────────────────────────────────────────────────────────────────────
// NGramLanguageClassifierTrainer  —  trains a character-n-gram naïve-Bayes language
// classifier from labelled text. The more (and the more representative) the text
// per language, the better it discriminates — including close Romance languages
// (es/pt/it/fr) on short input, where the small embedded detector struggles.
//
//   var t = new NGramLanguageClassifierTrainer();
//   t.Add("Italian", italianText).Add("Spanish", spanishText);   // or AddDirectory(...)
//   NGramLanguageClassifier model = t.Train();
//   model.Save("lang.bin");
// ─────────────────────────────────────────────────────────────────────────────

public sealed class NGramLanguageClassifierTrainer
{
    private readonly int _minN;
    private readonly int _maxN;
    private readonly double _alpha;
    private readonly Dictionary<string, Dictionary<string, long>> _counts = new(StringComparer.Ordinal);
    private readonly Dictionary<string, long> _totals = new(StringComparer.Ordinal);

    /// <param name="minN">Smallest n-gram order (default 1).</param>
    /// <param name="maxN">Largest n-gram order (default 4).</param>
    /// <param name="alpha">Additive (Lidstone) smoothing; smaller = sharper discrimination (default 0.1).</param>
    public NGramLanguageClassifierTrainer(int minN = 1, int maxN = 4, double alpha = 0.1)
    {
        if (minN < 1 || maxN < minN) throw new ArgumentException("Require 1 <= minN <= maxN.");
        if (alpha <= 0) throw new ArgumentException("alpha must be > 0.");
        _minN = minN;
        _maxN = maxN;
        _alpha = alpha;
    }

    /// <summary>Accumulates a training sample for <paramref name="language"/>.</summary>
    public NGramLanguageClassifierTrainer Add(string language, string text)
    {
        if (string.IsNullOrWhiteSpace(text)) return this;
        if (!_counts.TryGetValue(language, out var langCounts))
        {
            langCounts = new Dictionary<string, long>(StringComparer.Ordinal);
            _counts[language] = langCounts;
            _totals[language] = 0;
        }
        foreach (var (gram, k) in NGrams.Extract(text, _minN, _maxN))
        {
            langCounts[gram] = langCounts.GetValueOrDefault(gram) + k;
            _totals[language] += k;
        }
        return this;
    }

    /// <summary>Adds a text file's contents as a sample for <paramref name="language"/>.</summary>
    public NGramLanguageClassifierTrainer AddFile(string language, string path)
        => Add(language, File.ReadAllText(path));

    /// <summary>Adds every <c>*.txt</c> in <paramref name="directory"/>, using each file name (sans extension) as the language label.</summary>
    public NGramLanguageClassifierTrainer AddDirectory(string directory)
    {
        foreach (var file in Directory.EnumerateFiles(directory, "*.txt"))
            Add(Path.GetFileNameWithoutExtension(file), File.ReadAllText(file));
        return this;
    }

    /// <summary>Finalises the model: converts counts to smoothed log-probabilities (uniform language prior).</summary>
    public NGramLanguageClassifier Train()
    {
        if (_counts.Count == 0) throw new InvalidOperationException("No training data was added.");

        var languages = _counts.Keys.OrderBy(l => l, StringComparer.Ordinal).ToArray();

        var vocabulary = new HashSet<string>(StringComparer.Ordinal);
        foreach (var langCounts in _counts.Values)
            foreach (var gram in langCounts.Keys)
                vocabulary.Add(gram);
        int vocabSize = vocabulary.Count;

        double uniformLogPrior = Math.Log(1.0 / languages.Length);
        var logPrior = new double[languages.Length];
        var logProb = new Dictionary<string, float>[languages.Length];
        var fallback = new float[languages.Length];

        for (int i = 0; i < languages.Length; i++)
        {
            var langCounts = _counts[languages[i]];
            double denominator = _totals[languages[i]] + _alpha * vocabSize;

            logPrior[i] = uniformLogPrior;
            var lp = new Dictionary<string, float>(langCounts.Count, StringComparer.Ordinal);
            foreach (var (gram, k) in langCounts)
                lp[gram] = (float)Math.Log((k + _alpha) / denominator);
            logProb[i] = lp;
            fallback[i] = (float)Math.Log(_alpha / denominator);
        }

        return new NGramLanguageClassifier(_minN, _maxN, languages, logPrior, logProb, fallback);
    }
}
