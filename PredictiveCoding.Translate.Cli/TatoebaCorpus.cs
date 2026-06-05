using System.IO.Compression;

// ─────────────────────────────────────────────────────────────────────────────
// TatoebaCorpus  —  downloads real parallel data from OPUS-Tatoeba and builds the
// many-to-target training TSV (<tag>\t<source>\t<target>).
//
// For each requested language L it fetches the English↔L Moses bundle, then emits
// BOTH directions so one model learns many-to-target:
//     English → L   (tag = L)
//     L → English   (tag = eng)
// ─────────────────────────────────────────────────────────────────────────────

public static class TatoebaCorpus
{
    private const string OpusBase = "https://object.pouta.csc.fi/OPUS-Tatoeba/v2023-04-12/moses/";

    // OPUS uses 2-letter codes; the model trains on 3-letter tags (matching the
    // RIASSUNTAZZO library's LanguageTags map).
    private static readonly Dictionary<string, string> ThreeLetter = new(StringComparer.OrdinalIgnoreCase)
    {
        ["en"] = "eng", ["it"] = "ita", ["fr"] = "fra", ["es"] = "spa", ["de"] = "deu",
        ["pt"] = "por", ["nl"] = "nld", ["ru"] = "rus", ["pl"] = "pol", ["sv"] = "swe",
        ["da"] = "dan", ["fi"] = "fin", ["no"] = "nor", ["tr"] = "tur", ["el"] = "ell",
        ["cs"] = "ces", ["ro"] = "ron", ["hu"] = "hun", ["ar"] = "ara", ["he"] = "heb",
        ["ja"] = "jpn", ["zh"] = "zho", ["ko"] = "kor", ["uk"] = "ukr", ["ca"] = "cat",
    };

    public static string Tag(string twoLetter)
        => ThreeLetter.TryGetValue(twoLetter, out var t) ? t : twoLetter.ToLowerInvariant();

    /// <summary>Downloads/caches the requested pairs and writes the combined TSV; returns its path.</summary>
    public static async Task<string> PrepareAsync(
        string cacheDir, IReadOnlyList<string> languages, int maxPairsPerDirection, int maxChars, Action<string> log)
    {
        Directory.CreateDirectory(cacheDir);
        string tsvPath = Path.Combine(cacheDir, "tatoeba-many2target.tsv");

        using var writer = new StreamWriter(tsvPath, append: false);
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(10) };
        long total = 0;

        foreach (var raw in languages)
        {
            string l = raw.Trim().ToLowerInvariant();
            if (l.Length == 0 || l == "en") continue;

            string pair = string.CompareOrdinal("en", l) < 0 ? $"en-{l}" : $"{l}-en";
            string zip  = Path.Combine(cacheDir, $"{pair}.txt.zip");
            string dir  = Path.Combine(cacheDir, pair);

            if (!File.Exists(zip))
            {
                string url = OpusBase + pair + ".txt.zip";
                log($"[tatoeba] downloading {url}");
                try
                {
                    await using var src = await http.GetStreamAsync(url);
                    await using var dst = File.Create(zip);
                    await src.CopyToAsync(dst);
                }
                catch (Exception ex) { log($"[tatoeba]   skip {pair}: {ex.Message}"); continue; }
            }

            if (!Directory.Exists(dir))
            {
                Directory.CreateDirectory(dir);
                ZipFile.ExtractToDirectory(zip, dir, overwriteFiles: true);
            }

            string enFile = Path.Combine(dir, $"Tatoeba.{pair}.en");
            string foFile = Path.Combine(dir, $"Tatoeba.{pair}.{l}");
            if (!File.Exists(enFile) || !File.Exists(foFile)) { log($"[tatoeba]   missing aligned files for {pair}"); continue; }

            string tagFo = Tag(l);
            int written = 0;
            using var en = new StreamReader(enFile);
            using var fo = new StreamReader(foFile);
            while (written < maxPairsPerDirection)
            {
                string? e = await en.ReadLineAsync();
                string? f = await fo.ReadLineAsync();
                if (e is null || f is null) break;

                e = e.Trim().Replace('\t', ' ');
                f = f.Trim().Replace('\t', ' ');
                if (e.Length == 0 || f.Length == 0 || e.Length > maxChars || f.Length > maxChars) continue;

                writer.WriteLine($"{tagFo}\t{e}\t{f}");      // English → foreign
                writer.WriteLine($"eng\t{f}\t{e}");          // foreign → English
                written++;
                total += 2;
            }
            log($"[tatoeba] {pair}: {written} pairs × 2 directions");
        }

        writer.Flush();
        log($"[tatoeba] wrote {total:N0} examples to {tsvPath}");
        return tsvPath;
    }
}
