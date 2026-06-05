using System.Globalization;
using System.Text;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Biosignals;

/// <summary>
/// Reader for EDF (European Data Format) biosignal files — EEG, EMG, ECG, etc.
/// Parses the 256-byte main header, the per-signal headers and the int16 data
/// records, scaling each sample from digital to physical units.
/// </summary>
public static class EdfReader
{
    /// <summary>Reads and parses an EDF file from disk.</summary>
    public static EdfRecording ReadFromFile(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("EDF file not found.", path);
        return Parse(File.ReadAllBytes(path));
    }

    /// <summary>Parses EDF bytes into an <see cref="EdfRecording"/>.</summary>
    public static EdfRecording Parse(byte[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (data.Length < 256)
            throw new FormatException("EDF data is too short for a header.");

        string startDate = Ascii(data, 168, 8);
        string startTime = Ascii(data, 176, 8);
        int headerBytes = ParseInt(Ascii(data, 184, 8), "header length");
        int numRecords = ParseInt(Ascii(data, 236, 8), "record count");
        double recordDuration = ParseDouble(Ascii(data, 244, 8), "record duration");
        int signalCount = ParseInt(Ascii(data, 252, 4), "signal count");

        if (signalCount <= 0)
            throw new FormatException("EDF declares no signals.");

        int fullHeader = 256 + signalCount * 256;
        if (data.Length < fullHeader)
            throw new FormatException("EDF signal headers are truncated.");

        int offset = 256;
        string[] labels = ReadFields(data, ref offset, signalCount, 16);
        offset += signalCount * 80; // transducer type
        string[] physicalDims = ReadFields(data, ref offset, signalCount, 8);
        double[] physicalMin = ReadDoubles(data, ref offset, signalCount, 8, "physical minimum");
        double[] physicalMax = ReadDoubles(data, ref offset, signalCount, 8, "physical maximum");
        double[] digitalMin = ReadDoubles(data, ref offset, signalCount, 8, "digital minimum");
        double[] digitalMax = ReadDoubles(data, ref offset, signalCount, 8, "digital maximum");
        offset += signalCount * 80; // prefiltering
        int[] samplesPerRecord = ReadInts(data, ref offset, signalCount, 8, "samples per record");
        offset += signalCount * 32; // reserved

        int dataOffset = headerBytes > 0 ? headerBytes : fullHeader;

        int samplesPerRecordTotal = 0;
        foreach (int n in samplesPerRecord)
            samplesPerRecordTotal += n;
        if (samplesPerRecordTotal <= 0)
            throw new FormatException("EDF signals declare no samples.");

        if (numRecords < 0)
            numRecords = (data.Length - dataOffset) / (2 * samplesPerRecordTotal);
        if (numRecords <= 0)
            throw new FormatException("EDF contains no data records.");

        var buffers = new float[signalCount][];
        for (int s = 0; s < signalCount; s++)
        {
            int length = numRecords * samplesPerRecord[s];
            buffers[s] = new float[length > 0 ? length : 1];
        }

        var cursors = new int[signalCount];
        int position = dataOffset;
        for (int record = 0; record < numRecords; record++)
        {
            for (int s = 0; s < signalCount; s++)
            {
                double range = digitalMax[s] - digitalMin[s];
                double scale = range != 0 ? (physicalMax[s] - physicalMin[s]) / range : 0;
                for (int k = 0; k < samplesPerRecord[s]; k++)
                {
                    if (position + 2 > data.Length)
                        throw new FormatException("EDF data records are truncated.");
                    short digital = (short)(data[position] | (data[position + 1] << 8));
                    position += 2;
                    buffers[s][cursors[s]++] = (float)((digital - digitalMin[s]) * scale + physicalMin[s]);
                }
            }
        }

        var signals = new EdfSignal[signalCount];
        for (int s = 0; s < signalCount; s++)
        {
            double rate = recordDuration > 0 ? samplesPerRecord[s] / recordDuration : 0;
            var tensor = new DenseTensor<float>(buffers[s], buffers[s].Length);
            signals[s] = new EdfSignal(labels[s], physicalDims[s], rate, tensor);
        }

        return new EdfRecording(ParseStart(startDate, startTime), signals);
    }

    // ── Header helpers ────────────────────────────────────────────────────────

    private static string Ascii(byte[] data, int offset, int length)
        => Encoding.ASCII.GetString(data, offset, length).Trim();

    private static string[] ReadFields(byte[] data, ref int offset, int count, int fieldLength)
    {
        var fields = new string[count];
        for (int i = 0; i < count; i++)
            fields[i] = Ascii(data, offset + i * fieldLength, fieldLength);
        offset += count * fieldLength;
        return fields;
    }

    private static double[] ReadDoubles(byte[] data, ref int offset, int count, int fieldLength, string what)
    {
        var values = new double[count];
        for (int i = 0; i < count; i++)
            values[i] = ParseDouble(Ascii(data, offset + i * fieldLength, fieldLength), what);
        offset += count * fieldLength;
        return values;
    }

    private static int[] ReadInts(byte[] data, ref int offset, int count, int fieldLength, string what)
    {
        var values = new int[count];
        for (int i = 0; i < count; i++)
            values[i] = ParseInt(Ascii(data, offset + i * fieldLength, fieldLength), what);
        offset += count * fieldLength;
        return values;
    }

    private static int ParseInt(string text, string what)
        => int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value)
            ? value
            : throw new FormatException($"Invalid EDF {what}: '{text}'.");

    private static double ParseDouble(string text, string what)
        => double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value)
            ? value
            : throw new FormatException($"Invalid EDF {what}: '{text}'.");

    private static DateTime? ParseStart(string date, string time)
    {
        string[] d = date.Split('.');
        string[] t = time.Split('.');
        if (d.Length != 3 || t.Length != 3)
            return null;
        if (!int.TryParse(d[0], out int day) || !int.TryParse(d[1], out int month) || !int.TryParse(d[2], out int year) ||
            !int.TryParse(t[0], out int hour) || !int.TryParse(t[1], out int minute) || !int.TryParse(t[2], out int second))
            return null;

        year += year >= 85 ? 1900 : 2000; // EDF clipping convention
        try
        {
            return new DateTime(year, month, day, hour, minute, second, DateTimeKind.Unspecified);
        }
        catch (ArgumentOutOfRangeException)
        {
            return null;
        }
    }
}
