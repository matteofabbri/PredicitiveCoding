using System.IO.Compression;
using System.Text;
using DocumentFormat.OpenXml;
using DocumentFormat.OpenXml.Packaging;
using DocumentFormat.OpenXml.Wordprocessing;
using PredictiveCoding.Data.Documents;
using UglyToad.PdfPig.Core;
using UglyToad.PdfPig.Fonts.Standard14Fonts;
using UglyToad.PdfPig.Writer;

namespace PredicitiveCoding.UnitTests;

/// <summary>Offline tests for the PDF, DOCX and EPUB document readers, using
/// library writers / a hand-built EPUB to produce fixtures.</summary>
public sealed class DocumentsTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"pcdata_doc_{Guid.NewGuid():N}");

    public DocumentsTests() => Directory.CreateDirectory(_tmp);

    public void Dispose()
    {
        if (Directory.Exists(_tmp)) Directory.Delete(_tmp, true);
    }

    // ── PDF ──────────────────────────────────────────────────────────────────

    [Fact]
    public void Pdf_Read_ExtractsText()
    {
        var builder = new PdfDocumentBuilder();
        PdfPageBuilder page = builder.AddPage(595, 842); // A4 in points
        PdfDocumentBuilder.AddedFont font = builder.AddStandard14Font(Standard14Font.Helvetica);
        page.AddText("HelloPdf", 12, new PdfPoint(25, 700), font);
        byte[] bytes = builder.Build();

        PdfDocumentText text = PdfText.Read(bytes);

        Assert.Equal(1, text.PageCount);
        Assert.Contains("HelloPdf", text.Text);
    }

    // ── DOCX ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Docx_Read_ExtractsParagraphs()
    {
        string path = Path.Combine(_tmp, "doc.docx");
        using (var document = WordprocessingDocument.Create(path, WordprocessingDocumentType.Document))
        {
            MainDocumentPart main = document.AddMainDocumentPart();
            main.Document = new Document(new Body(
                new Paragraph(new Run(new Text("First line"))),
                new Paragraph(new Run(new Text("Second line")))));
            main.Document.Save();
        }

        string text = DocxText.Read(path);

        Assert.Contains("First line", text);
        Assert.Contains("Second line", text);
    }

    // ── EPUB ─────────────────────────────────────────────────────────────────

    [Fact]
    public void Epub_Read_ExtractsTitleAndText()
    {
        string path = WriteMinimalEpub();

        EpubContent content = EpubText.Read(path);

        Assert.Equal("Test Book", content.Title);
        Assert.Contains("Test Author", content.Authors);
        Assert.Contains("Hello EPUB content", content.Text);
    }

    private string WriteMinimalEpub()
    {
        string path = Path.Combine(_tmp, "book.epub");
        using var fileStream = File.Create(path);
        using var archive = new ZipArchive(fileStream, ZipArchiveMode.Create);

        ZipArchiveEntry mimetype = archive.CreateEntry("mimetype", CompressionLevel.NoCompression);
        using (var stream = mimetype.Open())
        {
            byte[] bytes = Encoding.ASCII.GetBytes("application/epub+zip");
            stream.Write(bytes, 0, bytes.Length);
        }

        AddEntry(archive, "META-INF/container.xml", """
            <?xml version="1.0"?>
            <container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
              <rootfiles>
                <rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/>
              </rootfiles>
            </container>
            """);

        AddEntry(archive, "OEBPS/content.opf", """
            <?xml version="1.0" encoding="utf-8"?>
            <package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="bookid">
              <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
                <dc:identifier id="bookid">urn:uuid:00000000-0000-0000-0000-000000000001</dc:identifier>
                <dc:title>Test Book</dc:title>
                <dc:language>en</dc:language>
                <dc:creator>Test Author</dc:creator>
                <meta property="dcterms:modified">2026-01-01T00:00:00Z</meta>
              </metadata>
              <manifest>
                <item id="ch1" href="chapter1.xhtml" media-type="application/xhtml+xml"/>
                <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
              </manifest>
              <spine>
                <itemref idref="ch1"/>
              </spine>
            </package>
            """);

        AddEntry(archive, "OEBPS/nav.xhtml", """
            <?xml version="1.0" encoding="utf-8"?>
            <html xmlns="http://www.w3.org/1999/xhtml" xmlns:epub="http://www.idpf.org/2007/ops">
              <head><title>Nav</title></head>
              <body><nav epub:type="toc"><ol><li><a href="chapter1.xhtml">Chapter 1</a></li></ol></nav></body>
            </html>
            """);

        AddEntry(archive, "OEBPS/chapter1.xhtml", """
            <?xml version="1.0" encoding="utf-8"?>
            <html xmlns="http://www.w3.org/1999/xhtml">
              <head><title>Chapter 1</title></head>
              <body><p>Hello EPUB content</p></body>
            </html>
            """);

        return path;
    }

    private static void AddEntry(ZipArchive archive, string name, string content)
    {
        ZipArchiveEntry entry = archive.CreateEntry(name);
        using var stream = entry.Open();
        byte[] bytes = Encoding.UTF8.GetBytes(content);
        stream.Write(bytes, 0, bytes.Length);
    }
}
