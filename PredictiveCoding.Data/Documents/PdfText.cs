using UglyToad.PdfPig;
using UglyToad.PdfPig.Content;

namespace PredictiveCoding.Data.Documents;

/// <summary>Text extracted from a PDF, per page.</summary>
public sealed record PdfDocumentText(int PageCount, IReadOnlyList<string> Pages)
{
    /// <summary>All pages joined into a single string.</summary>
    public string Text => string.Join("\n\n", Pages);
}

/// <summary>
/// Extracts text from PDF files using PdfPig.
/// </summary>
public static class PdfText
{
    /// <summary>Extracts text from a PDF file.</summary>
    public static PdfDocumentText Read(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("PDF file not found.", path);
        using PdfDocument document = PdfDocument.Open(path);
        return Extract(document);
    }

    /// <summary>Extracts text from PDF bytes.</summary>
    public static PdfDocumentText Read(byte[] bytes)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        using PdfDocument document = PdfDocument.Open(bytes);
        return Extract(document);
    }

    private static PdfDocumentText Extract(PdfDocument document)
    {
        var pages = new List<string>();
        foreach (Page page in document.GetPages())
            pages.Add(page.Text);
        return new PdfDocumentText(pages.Count, pages);
    }
}
