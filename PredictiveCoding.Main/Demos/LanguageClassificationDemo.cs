using System.Text;
using PredictiveCoding.Core;
using PredictiveCoding.Data.Text;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Main.Demos;

// ─────────────────────────────────────────────────────────────────────────────
// LanguageClassificationDemo
//
// Bridges PredictiveCoding.Data with the core Predictive Coding network on a real
// text task: identifying the language of a sentence (7 European languages).
//
// Pipeline:
//   1. Corpus     — the embedded multilingual seed corpus shipped in
//                   PredictiveCoding.Data (LanguageSamples.Default()), the very
//                   same data the bundled naive-Bayes classifier trains on.
//   2. Features   — each sentence is turned into a fixed FEATURE_DIM vector by
//                   hashing its character n-grams (orders 2–4) into buckets and
//                   L2-normalising. This is the only hand-written step; it is the
//                   "sensory encoding" the PC network sees.
//   3. PC network — a hierarchical predictive-coding Network learns to classify
//                   languages by minimum free energy (argmin_c F(x, one-hot c)).
//   4. Baseline   — NGramLanguageClassifier.CreateDefault() from the Data library
//                   gives a strong character-n-gram naive-Bayes reference.
//
// The demo trains the PC network, then compares both models on short phrases that
// are NOT in the corpus, and finally shows the probability distribution the PC
// network derives from per-class free energy.  Fully offline and deterministic.
// ─────────────────────────────────────────────────────────────────────────────

public static class LanguageClassificationDemo
{
    private const int   FEATURE_DIM = 512;   // hashed character-n-gram buckets
    private const int   HIDDEN      = 96;
    private const int   INFER_STEPS = 20;
    private const int   EPOCHS      = 50;
    private const float STATE_LR    = 0.1f;
    private const float WEIGHT_LR   = 0.02f;
    private const int   SEED        = 42;

    // Two-letter display codes for the corpus languages.
    private static readonly Dictionary<string, string> Code = new(StringComparer.Ordinal)
    {
        ["English"] = "EN", ["Italian"] = "IT", ["French"] = "FR", ["Spanish"] = "ES",
        ["German"]  = "DE", ["Portuguese"] = "PT", ["Dutch"] = "NL",
    };

    // Short, hand-written test phrases (none appear in the training corpus).
    private static readonly (string Lang, string Text)[] TestPhrases =
    [
        ("English", "Where is the nearest train station, please?"),
        ("English", "I really enjoyed the film we watched last night."),
        ("Italian", "Dove posso trovare un buon ristorante qui vicino?"),
        ("Italian", "Domani mattina andiamo al mare con gli amici."),
        ("French",  "Pouvez-vous m'indiquer le chemin de la gare?"),
        ("French",  "J'aimerais réserver une table pour deux personnes."),
        ("Spanish", "¿Dónde puedo comprar billetes para el tren?"),
        ("Spanish", "Me gustaría tomar un café con leche, por favor."),
        ("German",  "Können Sie mir bitte den Weg zum Bahnhof zeigen?"),
        ("German",  "Ich möchte ein Zimmer für zwei Nächte buchen."),
        ("Portuguese", "Onde fica a estação de comboios mais próxima?"),
        ("Portuguese", "Gostaria de reservar uma mesa para esta noite."),
        ("Dutch",   "Kunt u mij de weg naar het station wijzen?"),
        ("Dutch",   "Ik wil graag een kopje koffie bestellen, alstublieft."),
    ];

    public static void Run()
    {
        Console.WriteLine();
        Console.WriteLine("╔══════════════════════════════════════════════════════╗");
        Console.WriteLine("║    Language ID  ·  Predictive Coding vs naive-Bayes  ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════╝");
        Console.WriteLine();

        var rng = new Random(SEED);

        // ── 1. Corpus + label space (from PredictiveCoding.Data) ────────────────
        var corpus    = LanguageSamples.Default();
        var languages = LanguageSamples.Languages().ToArray();
        int numLangs  = languages.Length;
        var langIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < numLangs; i++) langIndex[languages[i]] = i;

        // ── 2. Baseline: bundled naive-Bayes classifier (also from Data) ────────
        var nb = NGramLanguageClassifier.CreateDefault();

        // ── 3. Featurise the corpus (with short word-window augmentation) ───────
        var train = new List<(int Label, DenseTensor<float> X)>();
        foreach (var (lang, text) in corpus)
        {
            int li = langIndex[lang];
            foreach (var fragment in Augment(text))
                train.Add((li, Featurize(fragment)));
        }

        Console.WriteLine($"  Languages   : {numLangs}  ({string.Join(", ", languages.Select(l => Code[l]))})");
        Console.WriteLine($"  Corpus      : {corpus.Count} sentences → {train.Count} training vectors (augmented)");
        Console.WriteLine($"  Features    : {FEATURE_DIM}-d hashed char n-grams (orders 2-4), L2-normalised");
        Console.WriteLine($"  PC network  : [{FEATURE_DIM}, {HIDDEN}, {numLangs}]  Tanh  η_s={STATE_LR}  η_w={WEIGHT_LR}  FFG warm-start");
        Console.WriteLine();

        string root = Path.Combine(Path.GetTempPath(), $"langpc_{Guid.NewGuid():N}");
        var cfg = new NetworkConfig
        {
            Path                   = root,
            LayerSizes             = [FEATURE_DIM, HIDDEN, numLangs],
            Activation             = ActivationType.Tanh,
            StateLearningRate      = STATE_LR,
            WeightLearningRate     = WEIGHT_LR,
            UseFeedforwardGistInit = true,
        };

        try
        {
            using var net = new Network(cfg, new Random(SEED));

            // ── 4. Train the PC network ─────────────────────────────────────────
            Console.WriteLine("  Training the Predictive Coding network...");
            Console.WriteLine();
            Console.WriteLine($"  {"Epoch",5}  {"Free energy",12}  {"Train acc",10}");
            Console.WriteLine("  " + new string('─', 33));

            for (int epoch = 1; epoch <= EPOCHS; epoch++)
            {
                Shuffle(train, rng);
                float sumE = 0f;
                foreach (var (li, x) in train)
                {
                    net.Learn(x, OneHot(li, numLangs), INFER_STEPS);
                    sumE += net.ComputeEnergy();
                }

                if (epoch == 1 || epoch % 5 == 0)
                {
                    float trainAcc = Accuracy(net, corpus, langIndex);
                    Console.WriteLine($"  {epoch,5}  {sumE / train.Count,12:F4}  {trainAcc,10:P1}");
                }
            }

            // ── 5. Evaluate both models on unseen short phrases ─────────────────
            Console.WriteLine();
            Console.WriteLine($"  Held-out phrases (not in corpus) — PC network vs naive-Bayes:");
            Console.WriteLine();
            Console.WriteLine($"  {"Phrase",-36} {"True",-5} {"PC net",-9} {"n-Bayes",-9}");
            Console.WriteLine("  " + new string('─', 62));

            int pcCorrect = 0, nbCorrect = 0;
            foreach (var (lang, text) in TestPhrases)
            {
                int trueIdx = langIndex[lang];
                var x       = Featurize(text);

                int pcIdx = net.Predict(x, INFER_STEPS);
                int nbIdx = langIndex.GetValueOrDefault(nb.Predict(text), -1);

                bool pcOk = pcIdx == trueIdx;
                bool nbOk = nbIdx == trueIdx;
                if (pcOk) pcCorrect++;
                if (nbOk) nbCorrect++;

                Console.WriteLine(
                    $"  {Truncate(text, 36),-36} {Code[lang],-5} " +
                    $"{CodeAt(languages, pcIdx) + (pcOk ? " ✓" : " ✗"),-9} " +
                    $"{CodeAt(languages, nbIdx) + (nbOk ? " ✓" : " ✗"),-9}");
            }

            int n = TestPhrases.Length;
            Console.WriteLine("  " + new string('─', 62));
            Console.WriteLine($"  Accuracy: PC network {100f * pcCorrect / n,5:F1}%   ·   naive-Bayes {100f * nbCorrect / n,5:F1}%");

            // ── 6. Free-energy → probability distribution (PC network) ──────────
            var (exLang, exText) = TestPhrases.First(p => p.Lang == "Spanish");
            Console.WriteLine();
            Console.WriteLine($"  How the PC network explains: \"{exText}\"  (true: {exLang})");
            Console.WriteLine();

            var energies = net.ClassEnergies(Featurize(exText), INFER_STEPS);
            var probs    = SoftmaxFromEnergies(energies);

            Console.WriteLine($"    {"Language",-12} {"Free energy",12}  {"P(lang | text)",16}");
            Console.WriteLine("    " + new string('─', 44));
            foreach (int i in Enumerable.Range(0, numLangs).OrderBy(i => energies[i]).Take(4))
                Console.WriteLine($"    {languages[i],-12} {energies[i],12:F4}  {probs[i],15:P1}");

            Console.WriteLine();
            Console.WriteLine("    naive-Bayes ranking (for comparison):");
            foreach (var d in nb.PredictAll(exText).Take(4))
                Console.WriteLine($"    {d.Language,-12} {"",12}  {d.Probability,15:P1}");
        }
        finally
        {
            if (Directory.Exists(root))
                Directory.Delete(root, recursive: true);
        }
    }

    // ── Featurisation ───────────────────────────────────────────────────────────

    /// <summary>
    /// Encodes text as a fixed FEATURE_DIM vector by hashing character n-grams
    /// (orders 2-4) of the normalised text into buckets, then L2-normalising so
    /// long and short inputs share the same scale (important for PC stability).
    /// </summary>
    private static DenseTensor<float> Featurize(string text)
    {
        var v = DenseTensor<float>.Zeros(FEATURE_DIM);
        string s = Normalize(text);

        for (int order = 2; order <= 4; order++)
            for (int i = 0; i + order <= s.Length; i++)
            {
                uint  h    = Fnv1a(s.AsSpan(i, order));
                int   idx  = (int)(h % FEATURE_DIM);
                float sign = ((h >> 16) & 1u) == 0 ? 1f : -1f;   // signed hashing reduces collision bias
                v.Buffer[idx] += sign;
            }

        float norm = 0f;
        for (int i = 0; i < FEATURE_DIM; i++) norm += v.Buffer[i] * v.Buffer[i];
        norm = MathF.Sqrt(norm);
        if (norm > 1e-8f)
            for (int i = 0; i < FEATURE_DIM; i++) v.Buffer[i] /= norm;

        return v;
    }

    /// <summary>Lower-case, keep letters (accents included), collapse runs of non-letters to a single space, wrap in spaces.</summary>
    private static string Normalize(string text)
    {
        var sb = new StringBuilder(text.Length + 2);
        sb.Append(' ');
        bool space = true;
        foreach (char c in text)
        {
            if (char.IsLetter(c)) { sb.Append(char.ToLowerInvariant(c)); space = false; }
            else if (!space)      { sb.Append(' '); space = true; }
        }
        if (!space) sb.Append(' ');
        return sb.ToString();
    }

    private static uint Fnv1a(ReadOnlySpan<char> s)
    {
        uint h = 2166136261u;
        foreach (char c in s) { h ^= c; h *= 16777619u; }
        return h;
    }

    /// <summary>The full sentence plus overlapping 5-word windows, so the model also sees short fragments.</summary>
    private static IEnumerable<string> Augment(string text)
    {
        yield return text;

        var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        const int window = 5, step = 3;
        for (int i = 0; i + window <= words.Length; i += step)
            yield return string.Join(' ', words[i..(i + window)]);
    }

    // ── Helpers ─────────────────────────────────────────────────────────────────

    private static float Accuracy(
        Network net,
        IReadOnlyList<(string Lang, string Text)> corpus,
        Dictionary<string, int> langIndex)
    {
        int correct = 0;
        foreach (var (lang, text) in corpus)
            if (net.Predict(Featurize(text), INFER_STEPS) == langIndex[lang]) correct++;
        return (float)correct / corpus.Count;
    }

    private static DenseTensor<float> OneHot(int index, int size)
    {
        var t = DenseTensor<float>.Zeros(size);
        t.Buffer[index] = 1f;
        return t;
    }

    /// <summary>P(lang) ∝ exp(−(F_lang − min F)): lower free energy ⇒ higher probability.</summary>
    private static double[] SoftmaxFromEnergies(float[] energies)
    {
        float min = energies.Min();
        var   p   = new double[energies.Length];
        double sum = 0;
        for (int i = 0; i < energies.Length; i++) { p[i] = Math.Exp(-(energies[i] - min)); sum += p[i]; }
        for (int i = 0; i < energies.Length; i++) p[i] /= sum;
        return p;
    }

    private static void Shuffle<T>(IList<T> list, Random rng)
    {
        for (int i = list.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (list[i], list[j]) = (list[j], list[i]);
        }
    }

    private static string CodeAt(string[] languages, int index)
        => index < 0 ? "??" : Code[languages[index]];

    private static string Truncate(string s, int max)
        => s.Length <= max ? s : s[..(max - 1)] + "…";
}
