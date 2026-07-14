namespace PredictiveCoding.Imaging;

/// <summary>
/// In-house Netpbm decoder: P2/P5 (grayscale, e.g. the AT&amp;T/ORL <c>.pgm</c>
/// faces) and P3/P6 (RGB <c>.ppm</c>), ASCII and binary, maxval up to 65535.
/// </summary>
internal static class NetpbmDecoder
{
    public static bool IsNetpbm(byte[] data)
        => data.Length > 2 && data[0] == 'P' && data[1] is >= (byte)'2' and <= (byte)'6' &&
           data[1] != '4';   // P4 (1-bit PBM) is not used by any dataset here

    public static RgbImage Decode(byte[] data)
    {
        int pos = 2;                       // past "P<n>"
        char kind = (char)data[1];
        int width = ReadValue(data, ref pos);
        int height = ReadValue(data, ref pos);
        int maxVal = ReadValue(data, ref pos);
        if (width < 1 || height < 1 || maxVal < 1)
            throw new InvalidDataException($"Invalid Netpbm header ({width}×{height}, maxval {maxVal}).");

        bool gray = kind is '2' or '5';
        bool ascii = kind is '2' or '3';
        int samples = width * height * (gray ? 1 : 3);

        var raw = new int[samples];
        if (ascii)
        {
            for (int i = 0; i < samples; i++)
                raw[i] = ReadValue(data, ref pos);
        }
        else
        {
            pos++;                          // single whitespace after maxval
            bool wide = maxVal > 255;       // two-byte big-endian samples
            int need = samples * (wide ? 2 : 1);
            if (data.Length - pos < need)
                throw new InvalidDataException("Netpbm pixel data truncated.");
            for (int i = 0; i < samples; i++)
                raw[i] = wide ? (data[pos + i * 2] << 8) | data[pos + i * 2 + 1] : data[pos + i];
        }

        var pixels = new byte[width * height * 3];
        for (int i = 0; i < width * height; i++)
        {
            if (gray)
            {
                byte v = Scale(raw[i], maxVal);
                pixels[i * 3] = v;
                pixels[i * 3 + 1] = v;
                pixels[i * 3 + 2] = v;
            }
            else
            {
                pixels[i * 3] = Scale(raw[i * 3], maxVal);
                pixels[i * 3 + 1] = Scale(raw[i * 3 + 1], maxVal);
                pixels[i * 3 + 2] = Scale(raw[i * 3 + 2], maxVal);
            }
        }
        return RgbImage.FromPixelData(pixels, width, height);
    }

    private static byte Scale(int value, int maxVal)
        => maxVal == 255 ? (byte)Math.Clamp(value, 0, 255)
                         : (byte)Math.Clamp(value * 255 / maxVal, 0, 255);

    // Reads the next integer, skipping whitespace and '#' comments.
    private static int ReadValue(byte[] data, ref int pos)
    {
        while (pos < data.Length)
        {
            byte b = data[pos];
            if (b == '#')
            {
                while (pos < data.Length && data[pos] != '\n') pos++;
            }
            else if (b is (byte)' ' or (byte)'\t' or (byte)'\r' or (byte)'\n')
            {
                pos++;
            }
            else
            {
                break;
            }
        }

        int start = pos;
        int value = 0;
        while (pos < data.Length && data[pos] is >= (byte)'0' and <= (byte)'9')
        {
            value = value * 10 + (data[pos] - '0');
            pos++;
        }
        if (pos == start)
            throw new InvalidDataException("Malformed Netpbm header.");
        return value;
    }
}
