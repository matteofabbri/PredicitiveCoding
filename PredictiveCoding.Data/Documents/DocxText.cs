using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;

namespace PredictiveCoding.Data.Documents;

/// <summary>
/// Extracts text from Word (.docx) files using the Open XML SDK.
/// </summary>
public static class DocxText
{
    /// <summary>Extracts the text of a .docx file (paragraphs joined by newlines).</summary>
    public static string Read(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("DOCX file not found.", path);
        using WordprocessingDocument document = WordprocessingDocument.Open(path, false);
        return Extract(document);
    }

    /// <summary>Extracts the text of a .docx stream.</summary>
    public static string Read(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using WordprocessingDocument document = WordprocessingDocument.Open(stream, false);
        return Extract(document);
    }

    private static string Extract(WordprocessingDocument document)
    {
        Body? body = document.MainDocumentPart?.Document?.Body;
        if (body is null)
            return string.Empty;
        return string.Join("\n", body.Descendants<Paragraph>().Select(p => p.InnerText));
    }
}
