namespace PredictiveCoding.Data.Text;

// ─────────────────────────────────────────────────────────────────────────────
// LanguageInfo  —  one row of the ISO-639 language table (Resources/languages.xml),
// imported with the language detector from Maddalena.Cognitive. LanguageDetector
// uses it to turn profile codes ("it") into names ("Italian") and full metadata.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>ISO-639 metadata for one language, as listed in the embedded table.</summary>
/// <param name="Family">Language family (e.g. "Indo-European").</param>
/// <param name="Name">English name as printed in the table — may list variants ("Dutch, Flemish").</param>
/// <param name="NativeName">Endonym (e.g. "Italiano").</param>
/// <param name="Iso6391">Two-letter ISO 639-1 code (e.g. "it").</param>
/// <param name="Iso6392T">ISO 639-2/T code.</param>
/// <param name="Iso6392B">ISO 639-2/B code.</param>
/// <param name="Iso6393">ISO 639-3 code.</param>
/// <param name="Notes">Free-form notes from the table.</param>
public sealed record LanguageInfo(
    string Family,
    string Name,
    string NativeName,
    string Iso6391,
    string Iso6392T,
    string Iso6392B,
    string Iso6393,
    string Notes);
