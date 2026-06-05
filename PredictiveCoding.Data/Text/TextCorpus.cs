namespace PredictiveCoding.Data.Text;

/// <summary>A text document: its source path and full content.</summary>
public sealed record TextDocument(string Path, string Text);

/// <summary>
/// Reads plain-text corpora from files and directories — the simplest source of
/// raw text for the LLM tokenizers.
/// </summary>
public static class TextCorpus
{
    /// <summary>Reads a single text file.</summary>
    public static TextDocument ReadFile(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("Text file not found.", path);
        return new TextDocument(path, File.ReadAllText(path));
    }

    /// <summary>Reads every matching text file in a directory (sorted by path).</summary>
    public static IReadOnlyList<TextDocument> ReadDirectory(
        string directory, string searchPattern = "*.txt", bool recursive = false)
        => [.. Enumerate(directory, searchPattern, recursive)];

    /// <summary>Lazily streams matching text files in a directory (sorted by path).</summary>
    public static IEnumerable<TextDocument> Enumerate(
        string directory, string searchPattern = "*.txt", bool recursive = false)
    {
        ArgumentException.ThrowIfNullOrEmpty(directory);
        if (!Directory.Exists(directory))
            throw new DirectoryNotFoundException($"Directory not found: '{directory}'.");

        SearchOption option = recursive ? SearchOption.AllDirectories : SearchOption.TopDirectoryOnly;
        foreach (string path in Directory.EnumerateFiles(directory, searchPattern, option).OrderBy(p => p, StringComparer.Ordinal))
            yield return new TextDocument(path, File.ReadAllText(path));
    }
}
