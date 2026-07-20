using System.Xml.Serialization;

namespace PredictiveCoding.Data.Text;

// ─────────────────────────────────────────────────────────────────────────────
// LanguageDetector  —  offline language detection for corpus routing/filtering.
//
// The real langdetect (Nakatani-Shuyo n-gram naïve Bayes) with the original 53
// Wikipedia-trained profiles, via Pēteris Ņikiforovs' C# port (Apache 2.0) —
// imported 2026-07-20 from Maddalena.Cognitive, where it ran in production. Files
// were copied in (Resources/LangProfiles, Resources/languages.xml), no reference
// to the old project; the only code change is Newtonsoft → System.Text.Json.
//
// It replaces the previous port of Catalyst's detector: the Catalyst model hashes
// its n-grams and loses the word-boundary information the string-keyed profiles
// keep, and it routinely mislabelled short Romance text — clean ~200-char Italian
// paragraphs came back "English" (verified against the real Catalyst NuGet, which
// fails the same way — it was the upstream model, not the port).
//
//   var det = LanguageDetector.CreateDefault(seed: 42);
//   string lang = det.Detect("Ciao, come stai?");            // "Italian"
//   foreach (var c in det.DetectAll(text)) { ... }           // ranked + probabilities
//   LanguageInfo? info = det.DetectInfo(text);               // full ISO metadata
// ─────────────────────────────────────────────────────────────────────────────

public sealed class LanguageDetector
{
    private static readonly Lazy<LanguageInfo[]> IsoTable = new(LoadIsoTable);

    // Name fixes the ISO table can't provide: its "ro" name cell is truncated to
    // "Moldavian", and the Chinese profiles use region codes the table doesn't list.
    private static readonly Dictionary<string, string> NameOverrides = new(StringComparer.Ordinal)
    {
        ["ro"] = "Romanian",
        ["zh-cn"] = "Chinese",
        ["zh-tw"] = "Chinese",
    };

    private readonly LanguageDetectorInternal _core;

    private LanguageDetector(LanguageDetectorInternal core) => _core = core;

    /// <summary>Detector backed by the 53 embedded langdetect profiles.</summary>
    /// <param name="seed">Optional RNG seed for reproducible results (detection is randomised).</param>
    public static LanguageDetector CreateDefault(int? seed = null)
        => new(new LanguageDetectorInternal { RandomSeed = seed });

    /// <summary>The full ISO-639 table shipped with the detector (metadata only —
    /// detection covers the 53 languages that have embedded profiles).</summary>
    public static IReadOnlyList<LanguageInfo> Languages => IsoTable.Value;

    /// <summary>
    /// Most probable language name ("Italian", "English", …), or "Unknown" when no
    /// language scores above threshold (empty input, or a script no profile covers).
    /// </summary>
    public string Detect(string text)
    {
        string? code = _core.Detect(text);
        return code is null ? "Unknown" : NameFor(code);
    }

    /// <summary>All candidate languages above threshold, highest probability first, by name.</summary>
    public IReadOnlyList<DetectedLanguage> DetectAll(string text)
    {
        var ranked = _core.DetectAll(text);
        var result = new DetectedLanguage[ranked.Count];
        for (int i = 0; i < ranked.Count; i++)
            result[i] = new DetectedLanguage(NameFor(ranked[i].Code), ranked[i].Probability);
        return result;
    }

    /// <summary>Full ISO metadata of the winning language, or null when nothing scores above threshold.</summary>
    public LanguageInfo? DetectInfo(string text)
    {
        string? code = _core.Detect(text);
        return code is null ? null : InfoFor(code);
    }

    // ── code → name/metadata ────────────────────────────────────────────────────

    private static string NameFor(string code)
    {
        if (NameOverrides.TryGetValue(code, out string? overridden)) return overridden;
        LanguageInfo? info = InfoFor(code);
        return info is null ? code : PrimaryName(info.Name);
    }

    private static LanguageInfo? InfoFor(string code)
    {
        // Region-specific profile codes ("zh-cn") fall back to the macro code ("zh").
        int dash = code.IndexOf('-');
        string iso = dash > 0 ? code[..dash] : code;
        return Array.Find(IsoTable.Value, l => string.Equals(l.Iso6391, iso, StringComparison.OrdinalIgnoreCase));
    }

    // "Dutch, Flemish" → "Dutch";  "Greek, Modern (1453-)" → "Greek".
    private static string PrimaryName(string isoName)
    {
        int comma = isoName.IndexOf(',');
        string name = comma >= 0 ? isoName[..comma] : isoName;
        int paren = name.IndexOf('(');
        if (paren >= 0) name = name[..paren];
        return name.Trim();
    }

    private static LanguageInfo[] LoadIsoTable()
    {
        var assembly = typeof(LanguageDetector).Assembly;
        string name = Array.Find(assembly.GetManifestResourceNames(),
                          n => n.EndsWith("languages.xml", StringComparison.Ordinal))
                      ?? throw new InvalidOperationException("Embedded 'languages.xml' not found.");

        using var stream = assembly.GetManifestResourceStream(name)!;
        var rows = (List<string[]>)new XmlSerializer(typeof(List<string[]>)).Deserialize(stream)!;

        // First row is the column header; keep only complete rows with a 639-1 code.
        return rows
            .Skip(1)
            .Where(r => r.Length >= 8 && !string.IsNullOrWhiteSpace(r[3]))
            .Select(r => new LanguageInfo(
                Family: r[0].Trim(), Name: r[1].Trim(), NativeName: r[2].Trim(),
                Iso6391: r[3].Trim(), Iso6392T: r[4].Trim(), Iso6392B: r[5].Trim(),
                Iso6393: r[6].Trim(), Notes: r[7].Trim()))
            .ToArray();
    }
}
