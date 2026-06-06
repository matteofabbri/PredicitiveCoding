using System.Text;
using PredictiveCoding.Data.Biosignals;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Offline tests for the EDF reader: build a minimal valid EDF in memory and
/// parse it, asserting the digital-to-physical scaling and metadata.
/// </summary>
public sealed class EdfReaderTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"pcdata_edf_{Guid.NewGuid():N}");

    public EdfReaderTests() => Directory.CreateDirectory(_tmp);

    public void Dispose()
    {
        if (Directory.Exists(_tmp)) Directory.Delete(_tmp, true);
    }

    private static void PutAscii(byte[] buffer, int offset, int length, string text)
    {
        for (int i = 0; i < length; i++) buffer[offset + i] = (byte)' ';
        byte[] bytes = Encoding.ASCII.GetBytes(text);
        Array.Copy(bytes, 0, buffer, offset, Math.Min(bytes.Length, length));
    }

    // One signal, one data record, 4 samples. Physical -100..100 over digital
    // -200..200 => scale 0.5, so physical = 0.5 * digital.
    private static byte[] BuildEdf()
    {
        const int signals = 1;
        const int samplesPerRecord = 4;
        const int records = 1;
        int headerLength = 256 + signals * 256;
        var data = new byte[headerLength + records * signals * samplesPerRecord * 2];

        PutAscii(data, 0, 8, "0");
        PutAscii(data, 8, 80, "patient");
        PutAscii(data, 88, 80, "recording");
        PutAscii(data, 168, 8, "01.01.20");
        PutAscii(data, 176, 8, "00.00.00");
        PutAscii(data, 184, 8, headerLength.ToString());
        PutAscii(data, 236, 8, records.ToString());
        PutAscii(data, 244, 8, "1");          // record duration, seconds
        PutAscii(data, 252, 4, signals.ToString());

        int o = 256;
        PutAscii(data, o, 16, "EEG"); o += signals * 16;
        o += signals * 80;                    // transducer
        PutAscii(data, o, 8, "uV"); o += signals * 8;
        PutAscii(data, o, 8, "-100"); o += signals * 8;   // physical min
        PutAscii(data, o, 8, "100"); o += signals * 8;    // physical max
        PutAscii(data, o, 8, "-200"); o += signals * 8;   // digital min
        PutAscii(data, o, 8, "200"); o += signals * 8;    // digital max
        o += signals * 80;                    // prefiltering
        PutAscii(data, o, 8, samplesPerRecord.ToString()); o += signals * 8;
        o += signals * 32;                    // reserved

        short[] digital = [0, 100, -100, 200];
        int p = headerLength;
        foreach (short d in digital)
        {
            data[p] = (byte)(d & 0xFF);
            data[p + 1] = (byte)((d >> 8) & 0xFF);
            p += 2;
        }
        return data;
    }

    [Fact]
    public void Parse_ReadsSignalMetadataAndScalesSamples()
    {
        EdfRecording recording = EdfReader.Parse(BuildEdf());

        EdfSignal signal = Assert.Single(recording.Signals);
        Assert.Equal("EEG", signal.Label);
        Assert.Equal("uV", signal.PhysicalDimension);
        Assert.Equal(4.0, signal.SamplingRate, 6);
        Assert.Equal([0f, 50f, -50f, 100f], signal.Samples.Buffer);
        Assert.Equal(new DateTime(2020, 1, 1, 0, 0, 0), recording.StartTime);
    }

    [Fact]
    public void Signal_LookupByLabel_IsCaseInsensitive()
    {
        EdfRecording recording = EdfReader.Parse(BuildEdf());
        Assert.NotNull(recording.Signal("eeg"));
        Assert.Null(recording.Signal("ecg"));
    }

    [Fact]
    public void ReadFromFile_RoundTrips()
    {
        string path = Path.Combine(_tmp, "rec.edf");
        File.WriteAllBytes(path, BuildEdf());

        EdfRecording recording = EdfReader.ReadFromFile(path);
        Assert.Single(recording.Signals);
    }

    [Fact]
    public void Parse_TooShort_Throws()
        => Assert.Throws<FormatException>(() => EdfReader.Parse(new byte[100]));

    [Fact]
    public void ReadFromFile_Missing_Throws()
        => Assert.Throws<FileNotFoundException>(() => EdfReader.ReadFromFile(Path.Combine(_tmp, "nope.edf")));
}
