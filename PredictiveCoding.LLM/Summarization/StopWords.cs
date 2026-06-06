namespace PredictiveCoding.LLM.Summarization;

// ─────────────────────────────────────────────────────────────────────────────
// StopWords  —  compact, built-in stop-word lists (no external models).
//
// Only the high-frequency function words for the languages the project actually
// exercises (English, Italian) are bundled; this keeps the summariser dependency-
// free while removing the noise that a pure length heuristic lets through. For any
// other / unknown language the summariser falls back to a content-word heuristic,
// so it degrades gracefully rather than failing. Register more lists at runtime via
// <see cref="Register"/>.
// ─────────────────────────────────────────────────────────────────────────────

public static class StopWords
{
    private static readonly Dictionary<string, HashSet<string>> ByCode = new(StringComparer.OrdinalIgnoreCase)
    {
        ["en"] = new(StringComparer.Ordinal)
        {
            "a","an","and","are","as","at","be","been","being","but","by","can","could","did","do","does",
            "doing","done","for","from","had","has","have","having","he","her","here","hers","him","his",
            "how","i","if","in","into","is","it","its","just","me","more","most","my","no","nor","not","of",
            "off","on","once","only","or","other","our","ours","out","over","own","same","she","should","so",
            "some","such","than","that","the","their","theirs","them","then","there","these","they","this",
            "those","through","to","too","under","until","up","very","was","we","were","what","when","where",
            "which","while","who","whom","why","will","with","would","you","your","yours",
        },
        ["it"] = new(StringComparer.Ordinal)
        {
            "a","ad","ai","al","alla","alle","allo","agli","anche","ancora","aveva","avere","ha","hanno","ho",
            "che","chi","ci","co","col","come","con","contro","cui","da","dai","dal","dalla","dalle","dallo",
            "degli","dei","del","della","delle","dello","di","dove","due","e","ed","era","essere","fa","fra",
            "gli","i","il","in","io","la","le","lei","li","lo","loro","lui","ma","me","mi","ne","nei","nel",
            "nella","nelle","nello","noi","non","nostro","o","od","ogni","per","perché","perche","piu","più",
            "po","poco","qua","quale","quando","quanto","quasi","quella","quelle","quelli","quello","questa",
            "queste","questi","questo","qui","se","sei","senza","si","sia","siamo","siete","solo","sono","sta",
            "su","sua","sue","sui","sul","sulla","sulle","sullo","suo","tra","tu","tua","tue","tuo","tutti",
            "tutto","un","una","uno","vi","voi",
        },
    };

    /// <summary>The stop-word set for a language code (2- or 3-letter, case-insensitive), or empty if unknown/null.</summary>
    public static IReadOnlySet<string> For(string? languageCode)
        => languageCode is not null && ByCode.TryGetValue(Normalize(languageCode), out var set)
            ? set
            : Empty;

    /// <summary>True when a built-in or registered list exists for the code.</summary>
    public static bool Has(string? languageCode)
        => languageCode is not null && ByCode.ContainsKey(Normalize(languageCode));

    /// <summary>Adds or replaces a stop-word list for a language code at runtime.</summary>
    public static void Register(string languageCode, IEnumerable<string> words)
        => ByCode[Normalize(languageCode)] = new HashSet<string>(words, StringComparer.Ordinal);

    private static readonly HashSet<string> Empty = new(StringComparer.Ordinal);

    // Fold common 3-letter / verbose codes onto the bundled 2-letter keys.
    private static string Normalize(string code) => code.Trim().ToLowerInvariant() switch
    {
        "eng" or "english" => "en",
        "ita" or "italian" or "italiano" => "it",
        var c when c.Length >= 2 => c[..2],
        var c => c,
    };
}
