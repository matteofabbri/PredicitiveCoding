using System.Reflection;
using System.Text;

namespace PredictiveCoding.Data.Text;

// ─────────────────────────────────────────────────────────────────────────────
// LanguageSamples  —  access to the embedded multilingual seed corpus
// (Resources/language-samples.tsv) as labelled (language, text) pairs.
//
// This is the same corpus that NGramLanguageClassifier.CreateDefault() trains on.
// Exposing it lets callers reuse the labelled data to train *other* models — e.g.
// a Predictive Coding network — on exactly the same task, for apples-to-apples
// comparisons.
// ─────────────────────────────────────────────────────────────────────────────

public static class LanguageSamples
{
    private const string ResourceSuffix = "language-samples.tsv";

    /// <summary>
    /// Reads the bundled seed corpus as <c>(Language, Text)</c> pairs, in file order.
    /// Lines that are empty or start with '#' are ignored.
    /// </summary>
    public static IReadOnlyList<(string Language, string Text)> Default()
    {
        Assembly assembly = typeof(LanguageSamples).Assembly;
        string name = Array.Find(assembly.GetManifestResourceNames(),
                          n => n.EndsWith(ResourceSuffix, StringComparison.Ordinal))
                      ?? throw new InvalidOperationException($"Embedded '{ResourceSuffix}' not found.");

        var samples = new List<(string, string)>();
        using var stream = assembly.GetManifestResourceStream(name)!;
        using var reader = new StreamReader(stream, Encoding.UTF8);
        string? line;
        while ((line = reader.ReadLine()) is not null)
        {
            if (line.Length == 0 || line[0] == '#') continue;
            int tab = line.IndexOf('\t');
            if (tab <= 0 || tab >= line.Length - 1) continue;
            samples.Add((line[..tab].Trim(), line[(tab + 1)..]));
        }
        return samples;
    }

    /// <summary>The distinct language labels in the seed corpus, in first-seen order.</summary>
    public static IReadOnlyList<string> Languages()
    {
        var seen = new List<string>();
        var set = new HashSet<string>(StringComparer.Ordinal);
        foreach (var (lang, _) in Default())
            if (set.Add(lang)) seen.Add(lang);
        return seen;
    }
}
