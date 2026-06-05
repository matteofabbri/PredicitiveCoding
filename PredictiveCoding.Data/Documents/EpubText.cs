using AngleSharp.Html.Parser;
using VersOne.Epub;

namespace PredictiveCoding.Data.Documents;

/// <summary>Content extracted from an EPUB ebook.</summary>
public sealed record EpubContent(
    string? Title,
    IReadOnlyList<string> Authors,
    string Text,
    IReadOnlyList<string> ChaptersHtml);

/// <summary>
/// Reads EPUB ebooks with VersOne.Epub and flattens them to plain text
/// (chapter HTML stripped with AngleSharp).
/// </summary>
public static class EpubText
{
    private static readonly HtmlParser Parser = new();

    /// <summary>Reads an EPUB file.</summary>
    public static EpubContent Read(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("EPUB file not found.", path);
        return Convert(EpubReader.ReadBook(path));
    }

    private static EpubContent Convert(EpubBook book)
    {
        var chapters = new List<string>();
        var textParts = new List<string>();

        foreach (EpubLocalTextContentFile file in book.ReadingOrder)
        {
            string html = file.Content;
            chapters.Add(html);

            using var document = Parser.ParseDocument(html);
            string text = document.Body?.TextContent ?? string.Empty;
            textParts.Add(string.Join(' ', text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)));
        }

        IReadOnlyList<string> authors = book.AuthorList ?? [];
        return new EpubContent(book.Title, authors, string.Join("\n\n", textParts), chapters);
    }
}
