using FellowOakDicom;
using FellowOakDicom.Imaging;
using FellowOakDicom.IO.Buffer;
using MimeKit;
using Parquet.Serialization;
using PredictiveCoding.Data.Email;
using PredictiveCoding.Data.Medical;
using PredictiveCoding.Data.Tabular;

namespace PredicitiveCoding.UnitTests;

/// <summary>Offline tests for the batch-4 sources that need a generated fixture:
/// Email (.eml via MimeKit), DICOM (fo-dicom) and Parquet (Parquet.Net).</summary>
public sealed class Lotto4FixtureTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"pcdata_l4_{Guid.NewGuid():N}");

    public Lotto4FixtureTests() => Directory.CreateDirectory(_tmp);

    public void Dispose()
    {
        if (Directory.Exists(_tmp)) Directory.Delete(_tmp, true);
    }

    // ── Email ────────────────────────────────────────────────────────────────

    [Fact]
    public void Email_Read_ParsesHeadersAndBody()
    {
        string path = Path.Combine(_tmp, "message.eml");
        var message = new MimeMessage();
        message.From.Add(new MailboxAddress("Alice", "alice@example.com"));
        message.To.Add(new MailboxAddress("Bob", "bob@example.com"));
        message.Subject = "Greetings";
        message.Body = new TextPart("plain") { Text = "Hello body" };
        using (var stream = File.Create(path))
            message.WriteTo(stream);

        EmailMessage email = EmailReader.Read(path);

        Assert.Equal("Greetings", email.Subject);
        Assert.Contains("bob@example.com", email.To[0]);
        Assert.Contains("Hello body", email.TextBody);
    }

    // ── DICOM ────────────────────────────────────────────────────────────────

    [Fact]
    public void Dicom_Read_ExtractsPixels()
    {
        string path = Path.Combine(_tmp, "image.dcm");
        var dataset = new DicomDataset
        {
            { DicomTag.SOPClassUID, DicomUID.SecondaryCaptureImageStorage },
            { DicomTag.SOPInstanceUID, DicomUIDGenerator.GenerateDerivedFromUUID() },
            { DicomTag.PhotometricInterpretation, "MONOCHROME2" },
            { DicomTag.SamplesPerPixel, (ushort)1 },
            { DicomTag.Rows, (ushort)2 },
            { DicomTag.Columns, (ushort)2 },
            { DicomTag.BitsAllocated, (ushort)16 },
            { DicomTag.BitsStored, (ushort)16 },
            { DicomTag.HighBit, (ushort)15 },
            { DicomTag.PixelRepresentation, (ushort)0 },
        };

        ushort[] pixels = [10, 20, 30, 40];
        var bytes = new byte[pixels.Length * 2];
        Buffer.BlockCopy(pixels, 0, bytes, 0, bytes.Length);
        var pixelData = DicomPixelData.Create(dataset, true);
        pixelData.AddFrame(new MemoryByteBuffer(bytes));

        new DicomFile(dataset).Save(path);

        DicomImageData image = DicomReader.Read(path);

        Assert.Equal(2, image.Rows);
        Assert.Equal(2, image.Columns);
        Assert.Equal(1, image.Frames);
        Assert.Equal([10f, 20f, 30f, 40f], image.Pixels.Buffer);
    }

    // ── Parquet ──────────────────────────────────────────────────────────────

    private sealed class Row
    {
        public double X { get; set; }
        public double Y { get; set; }
    }

    [Fact]
    public async Task Parquet_Load_ReadsNumericColumns()
    {
        string path = Path.Combine(_tmp, "data.parquet");
        var rows = new List<Row> { new() { X = 1, Y = 2 }, new() { X = 3, Y = 4 } };
        await using (var stream = File.Create(path))
            await ParquetSerializer.SerializeAsync(rows, stream);

        TabularData table = ParquetLoader.Load(path);

        Assert.Equal(2, table.RowCount);
        Assert.Equal(2, table.ColumnCount);
        int x = table.Columns.ToList().IndexOf("X");
        int y = table.Columns.ToList().IndexOf("Y");
        Assert.True(x >= 0 && y >= 0);
        Assert.Equal(1f, table.Values[0, x]);
        Assert.Equal(4f, table.Values[1, y]);
    }
}
