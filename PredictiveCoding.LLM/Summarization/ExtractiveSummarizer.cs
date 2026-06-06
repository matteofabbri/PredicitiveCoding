using System.Text;

namespace PredictiveCoding.LLM.Summarization;

// ─────────────────────────────────────────────────────────────────────────────
// ExtractiveSummarizer  —  "our own" extractive summariser, fully self-contained.
//
// Same job as RIASSUNTAZZO's CatalystSummarizer (pick the most representative
// sentences, keep them in reading order, stay in the source language) but with NO
// external NLP dependency — only the BCL. PredictiveCoding stays lean and offline.
//
// Pipeline:
//   1. split text into sentences (punctuation scan with abbreviation/decimal guards);
//   2. tokenise each sentence into normalised content words (lower-cased, stop-words
//      removed when a list is available, else a script-agnostic heuristic);
//   3. build L2-normalised TF-IDF vectors, treating each sentence as a "document";
//   4. score sentences — centroid cosine (default), TextRank PageRank, or raw TF;
//   5. greedily keep the top sentences, skipping ones too similar to those already
//      kept (MMR-style de-dup), then emit them in original reading order.
//
// Why word-level tokenisation and not the byte-level Bpe/ByteTokenizer: term
// frequency needs whole words; BPE sub-word pieces would distort the counts.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class ExtractiveSummarizer
{
    private static readonly SummaryOptions DefaultOptions = new();

    /// <summary>Extractive summary of <paramref name="text"/> in its original language.</summary>
    public string Summarize(string text, SummaryOptions? options = null)
    {
        options ??= DefaultOptions;
        if (string.IsNullOrWhiteSpace(text)) return string.Empty;

        var sentences = SplitSentences(text, options.MinSentenceChars);
        if (sentences.Count <= 1) return text.Trim();

        var stop = StopWords.For(options.Language);

        // Tokenise once; map every distinct term to a dense id for compact vectors.
        var vocab = new Dictionary<string, int>();
        var perSentenceTerms = new List<List<int>>(sentences.Count);
        foreach (var s in sentences)
        {
            var ids = new List<int>();
            foreach (var w in Tokenize(s.Text, stop))
            {
                if (!vocab.TryGetValue(w, out int id)) vocab[w] = id = vocab.Count;
                ids.Add(id);
            }
            perSentenceTerms.Add(ids);
        }
        if (vocab.Count == 0) return sentences[0].Text;   // nothing scorable → lead sentence

        var vectors = BuildTfIdf(perSentenceTerms, vocab.Count);

        double[] scores = options.Method switch
        {
            SummaryMethod.TextRank      => TextRank(vectors),
            SummaryMethod.TermFrequency => TermFrequency(perSentenceTerms),
            _                           => Centroid(vectors),
        };

        int keep = HowManyToKeep(sentences.Count, options);
        var chosen = Select(scores, vectors, keep, options.RedundancyThreshold);

        chosen.Sort();   // reading order
        return string.Join(" ", chosen.Select(i => sentences[i].Text));
    }

    /// <summary>Convenience overload: keep the top <paramref name="maxSentences"/> sentences.</summary>
    public string Summarize(string text, int maxSentences)
        => Summarize(text, new SummaryOptions { MaxSentences = maxSentences });

    // ── Sentence segmentation ────────────────────────────────────────────────────

    private readonly record struct Sentence(string Text);

    private static List<Sentence> SplitSentences(string text, int minChars)
    {
        var result = new List<Sentence>();
        int start = 0;
        for (int i = 0; i < text.Length; i++)
        {
            if (!IsTerminator(text[i])) continue;
            // Consume a run of terminators ("...", "?!").
            int end = i;
            while (end + 1 < text.Length && IsTerminator(text[end + 1])) end++;

            if (IsRealBoundary(text, i, end))
            {
                AddSentence(result, text, start, end, minChars);
                start = end + 1;
            }
            i = end;
        }
        AddSentence(result, text, start, text.Length - 1, minChars);   // trailing fragment
        return result;
    }

    private static void AddSentence(List<Sentence> acc, string text, int begin, int endInclusive, int minChars)
    {
        if (endInclusive < begin) return;
        string s = text.Substring(begin, endInclusive - begin + 1).Trim();
        if (s.Length >= minChars) acc.Add(new Sentence(s));
    }

    private static bool IsTerminator(char c) => c is '.' or '!' or '?' or '…' or '。' or '！' or '？';

    // A terminator is a real sentence boundary unless it is a decimal point or part of an
    // abbreviation/initial, and the following text must look like a new sentence (uppercase,
    // opening quote, digit, or end-of-text) to reject mid-sentence dots.
    private static bool IsRealBoundary(string text, int dot, int end)
    {
        if (text[dot] == '.')
        {
            // Decimal: digit '.' digit  →  not a boundary.
            if (dot > 0 && dot + 1 < text.Length && char.IsDigit(text[dot - 1]) && char.IsDigit(text[dot + 1]))
                return false;
            if (IsAbbreviationBefore(text, dot)) return false;
        }

        int j = end + 1;
        while (j < text.Length && char.IsWhiteSpace(text[j])) j++;
        if (j >= text.Length) return true;                                   // end of text
        char next = text[j];
        return char.IsUpper(next) || char.IsDigit(next) || next is '"' or '\'' or '«' or '“' or '(' or '‘';
    }

    private static bool IsAbbreviationBefore(string text, int dot)
    {
        int e = dot - 1, b = e;
        while (b >= 0 && (char.IsLetter(text[b]) || text[b] == '.')) b--;
        b++;
        if (b > e) return false;
        string word = text[b..(e + 1)].Replace(".", "").ToLowerInvariant();
        return word.Length == 1 || Abbreviations.Contains(word);            // single letter = initial
    }

    private static readonly HashSet<string> Abbreviations = new(StringComparer.Ordinal)
    {
        // English
        "mr","mrs","ms","dr","prof","sr","jr","st","vs","etc","eg","ie","fig","vol","no","pp","op","ca","inc","ltd","co",
        // Italian
        "sig","sigra","dott","dottssa","profssa","ing","avv","geom","rag","egr","gent","spett","ecc","es","pag","art","cfr","tel","nn",
    };

    // ── Tokenisation ─────────────────────────────────────────────────────────────

    private static IEnumerable<string> Tokenize(string sentence, IReadOnlySet<string> stop)
    {
        var sb = new StringBuilder();
        foreach (char c in sentence)
        {
            if (char.IsLetterOrDigit(c) || (sb.Length > 0 && c is '\'' or '-' or '’'))
                sb.Append(char.ToLowerInvariant(c));
            else if (sb.Length > 0) { foreach (var w in Emit(sb, stop)) yield return w; }
        }
        if (sb.Length > 0) foreach (var w in Emit(sb, stop)) yield return w;
    }

    private static IEnumerable<string> Emit(StringBuilder sb, IReadOnlySet<string> stop)
    {
        string w = sb.ToString().Trim('\'', '-', '’');
        sb.Clear();
        if (IsContentWord(w) && !stop.Contains(w)) yield return w;
    }

    private static bool IsContentWord(string w)
    {
        if (w.Length < 2) return false;
        foreach (char c in w) if (char.IsLetter(c)) return true;
        return false;
    }

    // ── TF-IDF vectors (sentence = document) ─────────────────────────────────────

    // Each vector is a sparse term→weight map, L2-normalised so cosine == dot product.
    private static Dictionary<int, double>[] BuildTfIdf(List<List<int>> perSentenceTerms, int vocabSize)
    {
        int n = perSentenceTerms.Count;
        var df = new int[vocabSize];
        foreach (var terms in perSentenceTerms)
            foreach (int t in terms.Distinct()) df[t]++;

        var idf = new double[vocabSize];
        for (int t = 0; t < vocabSize; t++)
            idf[t] = Math.Log((1.0 + n) / (1.0 + df[t])) + 1.0;            // smoothed, always > 0

        var vectors = new Dictionary<int, double>[n];
        for (int i = 0; i < n; i++)
        {
            var tf = new Dictionary<int, double>();
            foreach (int t in perSentenceTerms[i]) tf[t] = tf.GetValueOrDefault(t) + 1.0;

            var vec = new Dictionary<int, double>(tf.Count);
            double norm = 0;
            foreach (var (t, f) in tf) { double w = f * idf[t]; vec[t] = w; norm += w * w; }

            norm = Math.Sqrt(norm);
            if (norm > 0) foreach (var t in vec.Keys.ToArray()) vec[t] /= norm;
            vectors[i] = vec;
        }
        return vectors;
    }

    private static double Cosine(Dictionary<int, double> a, Dictionary<int, double> b)
    {
        if (b.Count < a.Count) (a, b) = (b, a);                            // iterate the smaller map
        double dot = 0;
        foreach (var (t, wa) in a) if (b.TryGetValue(t, out double wb)) dot += wa * wb;
        return dot;                                                        // vectors are unit-length
    }

    // ── Scoring strategies ───────────────────────────────────────────────────────

    private static double[] Centroid(Dictionary<int, double>[] vectors)
    {
        var centroid = new Dictionary<int, double>();
        foreach (var v in vectors)
            foreach (var (t, w) in v) centroid[t] = centroid.GetValueOrDefault(t) + w;

        double norm = Math.Sqrt(centroid.Values.Sum(w => w * w));
        if (norm > 0) foreach (var t in centroid.Keys.ToArray()) centroid[t] /= norm;

        var scores = new double[vectors.Length];
        for (int i = 0; i < vectors.Length; i++) scores[i] = Cosine(vectors[i], centroid);
        return scores;
    }

    // PageRank over the (undirected) sentence-similarity graph — TextRank / LexRank.
    private static double[] TextRank(Dictionary<int, double>[] vectors, double damping = 0.85, int maxIter = 50)
    {
        int n = vectors.Length;
        var sim = new double[n][];
        var rowSum = new double[n];
        for (int i = 0; i < n; i++)
        {
            sim[i] = new double[n];
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                double s = Cosine(vectors[i], vectors[j]);
                sim[i][j] = s;
                rowSum[i] += s;
            }
        }

        var rank = new double[n];
        Array.Fill(rank, 1.0 / n);
        var next = new double[n];
        for (int iter = 0; iter < maxIter; iter++)
        {
            for (int i = 0; i < n; i++)
            {
                double inbound = 0;
                for (int j = 0; j < n; j++)
                    if (i != j && rowSum[j] > 0) inbound += sim[j][i] / rowSum[j] * rank[j];
                next[i] = (1 - damping) / n + damping * inbound;
            }
            double delta = 0;
            for (int i = 0; i < n; i++) { delta += Math.Abs(next[i] - rank[i]); rank[i] = next[i]; }
            if (delta < 1e-6) break;
        }
        return rank;
    }

    private static double[] TermFrequency(List<List<int>> perSentenceTerms)
    {
        var freq = new Dictionary<int, int>();
        foreach (var terms in perSentenceTerms)
            foreach (int t in terms) freq[t] = freq.GetValueOrDefault(t) + 1;
        int max = freq.Count == 0 ? 1 : freq.Values.Max();

        var scores = new double[perSentenceTerms.Count];
        for (int i = 0; i < perSentenceTerms.Count; i++)
        {
            var terms = perSentenceTerms[i];
            double sum = terms.Sum(t => (double)freq[t] / max);
            scores[i] = terms.Count == 0 ? 0 : sum / Math.Sqrt(terms.Count);   // length-normalised
        }
        return scores;
    }

    // ── Selection (top-K with MMR-style redundancy filter) ───────────────────────

    private static int HowManyToKeep(int sentenceCount, SummaryOptions o)
    {
        int byRatio = (int)Math.Ceiling(sentenceCount * o.Ratio);
        int keep = o.MaxSentences > 0 ? Math.Min(o.MaxSentences, byRatio) : byRatio;
        return Math.Clamp(keep, 1, sentenceCount);
    }

    private static List<int> Select(double[] scores, Dictionary<int, double>[] vectors, int keep, double redundancy)
    {
        var order = Enumerable.Range(0, scores.Length).OrderByDescending(i => scores[i]).ToList();
        var chosen = new List<int>(keep);
        foreach (int i in order)
        {
            if (chosen.Count >= keep) break;
            if (redundancy < 1.0 && chosen.Any(j => Cosine(vectors[i], vectors[j]) > redundancy))
                continue;                                                  // too similar to a kept sentence
            chosen.Add(i);
        }
        // If de-dup starved us, top up by score to honour `keep`.
        if (chosen.Count < keep)
            foreach (int i in order) { if (chosen.Count >= keep) break; if (!chosen.Contains(i)) chosen.Add(i); }
        return chosen;
    }
}
