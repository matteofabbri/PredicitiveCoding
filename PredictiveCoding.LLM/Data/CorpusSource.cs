using System.Text;

namespace PredictiveCoding.LLM.Data;

// ─────────────────────────────────────────────────────────────────────────────
// CorpusSource  —  a re-enumerable stream of text lines (roadmap-Pro Phase 4).
//
// Wraps files, folders and globs behind one lazy, streaming line source; mixes
// several sources with sampling weights; and layers cheap hygiene (line dedup,
// length filters) as decorators. Everything stays lazy — nothing is buffered —
// so it feeds StreamingTextDataset over corpora far larger than RAM.
//
// A source is a FACTORY of enumerables (Func → IEnumerable), so it can be walked
// more than once (training makes two passes: held-out split + training split).
// Byte/line counts accumulate into a shared Stats object as lines are pulled.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class CorpusSource
{
    private readonly Func<IEnumerable<string>> _factory;

    /// <summary>Human-readable description of the composed pipeline (for reports).</summary>
    public string Description { get; }

    /// <summary>Cumulative line/byte counts across every enumeration so far.</summary>
    public CorpusStats Stats { get; } = new();

    private CorpusSource(string description, Func<IEnumerable<string>> factory)
    {
        Description = description;
        _factory    = factory;
    }

    /// <summary>Enumerates the lines, tallying line and UTF-8 byte counts into <see cref="Stats"/>.</summary>
    public IEnumerable<string> Lines()
    {
        foreach (string line in _factory())
        {
            Stats.Add(line);
            yield return line;
        }
    }

    // ── Leaf sources ─────────────────────────────────────────────────────────

    /// <summary>An in-memory string, split on newlines.</summary>
    public static CorpusSource FromText(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        return new("text", () => text.Split('\n'));
    }

    /// <summary>A single UTF-8 file, streamed line by line.</summary>
    public static CorpusSource FromFile(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        return new($"file:{path}", () => File.ReadLines(path));
    }

    /// <summary>
    /// Every file matching <paramref name="searchPattern"/> under
    /// <paramref name="directory"/> (sorted by path for a deterministic order),
    /// concatenated line by line.
    /// </summary>
    public static CorpusSource FromFolder(string directory, string searchPattern = "*.txt", bool recursive = true)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        var option = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        return new($"folder:{directory}/{searchPattern}{(recursive ? " (recursive)" : "")}",
            () => EnumerateFiles(directory, searchPattern, option));
    }

    /// <summary>
    /// A glob relative to <paramref name="baseDirectory"/> — the last path segment
    /// is the file pattern (e.g. <c>"**"</c> handled as recursive <c>*</c>), the
    /// rest is the directory. Files are read in sorted order.
    /// </summary>
    public static CorpusSource FromGlob(string baseDirectory, string pattern)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(baseDirectory);
        ArgumentException.ThrowIfNullOrWhiteSpace(pattern);
        bool recursive = pattern.Contains("**");
        string filePattern = pattern.Replace("**", "*");
        var option = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        return new($"glob:{baseDirectory}/{pattern}",
            () => EnumerateFiles(baseDirectory, filePattern, option));
    }

    private static IEnumerable<string> EnumerateFiles(string dir, string pattern, SearchOption option)
    {
        if (!Directory.Exists(dir)) yield break;
        var files = Directory.GetFiles(dir, pattern, option);
        Array.Sort(files, StringComparer.Ordinal);            // deterministic order
        foreach (string file in files)
            foreach (string line in File.ReadLines(file))
                yield return line;
    }

    // ── Hygiene decorators ─────────────────────────────────────────────────────

    /// <summary>
    /// Drops repeated lines (exact match, by 64-bit content hash so the memory
    /// stays bounded by the number of DISTINCT lines, not their length). A rare
    /// hash collision would drop a distinct line — acceptable at corpus scale.
    /// </summary>
    public CorpusSource Deduplicated() =>
        new($"{Description}|dedup", () => Dedup(_factory()));

    private static IEnumerable<string> Dedup(IEnumerable<string> lines)
    {
        var seen = new HashSet<long>();
        foreach (string line in lines)
            if (seen.Add(Fnv64(line))) yield return line;
    }

    /// <summary>
    /// Keeps only lines whose length in characters is in
    /// [<paramref name="minChars"/>, <paramref name="maxChars"/>] — trims boilerplate
    /// (empty/near-empty lines) and pathological long lines.
    /// </summary>
    public CorpusSource WithLengthFilter(int minChars = 1, int maxChars = int.MaxValue)
    {
        if (minChars < 0) throw new ArgumentOutOfRangeException(nameof(minChars));
        if (maxChars < minChars) throw new ArgumentOutOfRangeException(nameof(maxChars));
        return new($"{Description}|len[{minChars},{maxChars}]",
            () => _factory().Where(l => l.Length >= minChars && l.Length <= maxChars));
    }

    /// <summary>
    /// Generic line filter (ROADMAP-Scale C2) — the composition point for any
    /// hygiene the library does not ship (e.g. a language filter built on
    /// PredictiveCoding.Data's detector, which this project cannot reference).
    /// <paramref name="label"/> keeps <see cref="Description"/> honest.
    /// </summary>
    public CorpusSource Where(Func<string, bool> predicate, string label)
    {
        ArgumentNullException.ThrowIfNull(predicate);
        ArgumentException.ThrowIfNullOrWhiteSpace(label);
        return new($"{Description}|{label}", () => _factory().Where(predicate));
    }

    /// <summary>
    /// Web-scale boilerplate filter (ROADMAP-Scale C2): drops lines with fewer
    /// than <paramref name="minWords"/> words — the signature of navigation
    /// menus, cookie banners and button labels in crawled shards. Pair with
    /// <see cref="Deduplicated"/>, which removes the blocks a site repeats on
    /// every page (headers, footers) regardless of their length.
    /// </summary>
    public CorpusSource WithBoilerplateStripped(int minWords = 5)
    {
        if (minWords < 1) throw new ArgumentOutOfRangeException(nameof(minWords));
        return new($"{Description}|boiler[{minWords}w]",
            () => _factory().Where(l => CountWords(l) >= minWords));
    }

    private static int CountWords(string line)
    {
        int words = 0; bool inWord = false;
        foreach (char c in line)
        {
            if (char.IsWhiteSpace(c)) inWord = false;
            else if (!inWord) { inWord = true; words++; }
        }
        return words;
    }

    // ── Mixing ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// Deterministically interleaves several sources in proportion to their
    /// weights (no RNG — at each step it pulls from the source whose emitted/weight
    /// ratio is lowest, so an 80/20 mix stays ≈80/20 throughout the stream). When
    /// a source is exhausted the rest keep going. Weights must be positive.
    /// </summary>
    public static CorpusSource Mix(params (CorpusSource source, double weight)[] parts)
    {
        if (parts is null || parts.Length == 0)
            throw new ArgumentException("At least one source is required.", nameof(parts));
        foreach (var (_, weight) in parts)
            if (weight <= 0d) throw new ArgumentOutOfRangeException(nameof(parts), "Weights must be positive.");

        string desc = "mix(" + string.Join(", ", parts.Select(p => $"{p.source.Description}×{p.weight:0.##}")) + ")";
        return new(desc, () => WeightedInterleave(parts));
    }

    private static IEnumerable<string> WeightedInterleave((CorpusSource source, double weight)[] parts)
    {
        var iterators = parts.Select(p => p.source.Lines().GetEnumerator()).ToArray();
        var weights   = parts.Select(p => p.weight).ToArray();
        var emitted   = new double[parts.Length];
        var alive     = new bool[parts.Length];
        int living    = 0;
        for (int i = 0; i < iterators.Length; i++) { alive[i] = true; living++; }

        try
        {
            while (living > 0)
            {
                // Pick the living source most "behind" its weighted quota.
                int pick = -1;
                double bestRatio = double.MaxValue;
                for (int i = 0; i < iterators.Length; i++)
                {
                    if (!alive[i]) continue;
                    double ratio = emitted[i] / weights[i];
                    if (ratio < bestRatio) { bestRatio = ratio; pick = i; }
                }

                if (iterators[pick].MoveNext())
                {
                    emitted[pick] += 1d;
                    yield return iterators[pick].Current;
                }
                else
                {
                    alive[pick] = false;
                    living--;
                }
            }
        }
        finally
        {
            foreach (var it in iterators) it.Dispose();
        }
    }

    // FNV-1a 64-bit over the UTF-8 bytes of a string.
    private static long Fnv64(string s)
    {
        const ulong offset = 14695981039346656037UL, prime = 1099511628211UL;
        ulong h = offset;
        foreach (byte b in Encoding.UTF8.GetBytes(s)) { h ^= b; h *= prime; }
        return unchecked((long)h);
    }
}

/// <summary>Running line/byte tallies for a <see cref="CorpusSource"/> enumeration.</summary>
public sealed class CorpusStats
{
    public long Lines { get; private set; }
    public long Bytes { get; private set; }

    internal void Add(string line)
    {
        Lines++;
        Bytes += Encoding.UTF8.GetByteCount(line) + 1;   // +1 for the newline the dataset re-adds
    }

    public override string ToString() => $"{Lines:N0} lines, {Bytes / 1024.0:N1} KB";
}
