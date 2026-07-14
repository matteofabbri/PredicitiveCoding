namespace PredictiveCoding.Imaging;

/// <summary>
/// In-house animated-GIF encoder (GIF89a). Uses one global 256-colour table —
/// exact when the frames hold ≤256 distinct colours, otherwise an RGB 3-3-2
/// quantisation — and emits LZW streams in the fixed-9-bit form (a clear code
/// before the table can grow), which every decoder accepts. Optimised for
/// correctness and zero dependencies, not for file size.
/// </summary>
public static class GifEncoder
{
    private const int MinCodeSize = 8;          // palette indices are 8-bit
    private const int ClearCode = 256;
    private const int EndCode = 257;
    private const int CodesPerClear = 250;      // stay below the 9→10 bit growth point

    public static void Save(string path, IReadOnlyList<RgbImage> frames, int delayCentiseconds = 10)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frames.Count == 0)
            throw new ArgumentException("At least one frame is required.", nameof(frames));
        int width = frames[0].Width, height = frames[0].Height;
        foreach (RgbImage f in frames)
            if (f.Width != width || f.Height != height)
                throw new ArgumentException("All frames must share the same dimensions.", nameof(frames));

        (byte[] palette, Dictionary<int, byte>? exact) = BuildPalette(frames);

        using FileStream fs = File.Create(path);
        using var w = new BinaryWriter(fs);

        w.Write("GIF89a"u8);
        w.Write((ushort)width);
        w.Write((ushort)height);
        w.Write((byte)0xF7);                    // GCT present, 8-bit colour, 256 entries
        w.Write((byte)0);                       // background colour index
        w.Write((byte)0);                       // pixel aspect ratio
        w.Write(palette);                       // 256 × RGB

        if (frames.Count > 1)
            WriteLoopExtension(w);

        foreach (RgbImage frame in frames)
        {
            // Graphic-control extension: frame delay, no transparency.
            w.Write((byte)0x21);
            w.Write((byte)0xF9);
            w.Write((byte)4);
            w.Write((byte)0);
            w.Write((ushort)Math.Clamp(delayCentiseconds, 0, ushort.MaxValue));
            w.Write((byte)0);
            w.Write((byte)0);

            // Image descriptor: full frame, no local colour table, not interlaced.
            w.Write((byte)0x2C);
            w.Write((ushort)0);
            w.Write((ushort)0);
            w.Write((ushort)width);
            w.Write((ushort)height);
            w.Write((byte)0);

            WriteLzwData(w, frame, exact);
        }

        w.Write((byte)0x3B);                    // trailer
    }

    private static (byte[] Palette, Dictionary<int, byte>? Exact) BuildPalette(IReadOnlyList<RgbImage> frames)
    {
        var exact = new Dictionary<int, byte>();
        foreach (RgbImage f in frames)
        {
            byte[] p = f.Pixels;
            for (int i = 0; i < p.Length; i += 3)
            {
                int key = (p[i] << 16) | (p[i + 1] << 8) | p[i + 2];
                if (!exact.ContainsKey(key))
                {
                    if (exact.Count == 256)
                    {
                        exact = null!;
                        goto quantised;
                    }
                    exact[key] = (byte)exact.Count;
                }
            }
        }

        var table = new byte[256 * 3];
        foreach ((int key, byte index) in exact)
        {
            table[index * 3] = (byte)(key >> 16);
            table[index * 3 + 1] = (byte)(key >> 8);
            table[index * 3 + 2] = (byte)key;
        }
        return (table, exact);

    quantised:
        // RGB 3-3-2: index = RRRGGGBB, entry = centre of each cell.
        var quantTable = new byte[256 * 3];
        for (int i = 0; i < 256; i++)
        {
            quantTable[i * 3] = (byte)((i >> 5) * 255 / 7);
            quantTable[i * 3 + 1] = (byte)(((i >> 2) & 7) * 255 / 7);
            quantTable[i * 3 + 2] = (byte)((i & 3) * 255 / 3);
        }
        return (quantTable, null);
    }

    private static void WriteLoopExtension(BinaryWriter w)
    {
        w.Write((byte)0x21);
        w.Write((byte)0xFF);
        w.Write((byte)11);
        w.Write("NETSCAPE2.0"u8);
        w.Write((byte)3);
        w.Write((byte)1);
        w.Write((ushort)0);                     // loop forever
        w.Write((byte)0);
    }

    private static void WriteLzwData(BinaryWriter w, RgbImage frame, Dictionary<int, byte>? exact)
    {
        w.Write((byte)MinCodeSize);

        var packer = new BitPacker(w);
        packer.WriteCode(ClearCode);
        int sinceClear = 0;

        byte[] p = frame.Pixels;
        for (int i = 0; i < p.Length; i += 3)
        {
            byte index;
            if (exact is not null)
            {
                index = exact[(p[i] << 16) | (p[i + 1] << 8) | p[i + 2]];
            }
            else
            {
                index = (byte)((p[i] & 0xE0) | ((p[i + 1] >> 3) & 0x1C) | (p[i + 2] >> 6));
            }

            packer.WriteCode(index);
            if (++sinceClear >= CodesPerClear)
            {
                packer.WriteCode(ClearCode);
                sinceClear = 0;
            }
        }

        packer.WriteCode(EndCode);
        packer.Finish();
    }

    /// <summary>Packs 9-bit LZW codes LSB-first into 255-byte GIF sub-blocks.</summary>
    private sealed class BitPacker(BinaryWriter writer)
    {
        private readonly byte[] _block = new byte[255];
        private int _blockLen;
        private int _bitBuffer;
        private int _bitCount;

        public void WriteCode(int code)
        {
            _bitBuffer |= code << _bitCount;
            _bitCount += 9;
            while (_bitCount >= 8)
            {
                AddByte((byte)_bitBuffer);
                _bitBuffer >>= 8;
                _bitCount -= 8;
            }
        }

        public void Finish()
        {
            if (_bitCount > 0)
                AddByte((byte)_bitBuffer);
            FlushBlock();
            writer.Write((byte)0);              // block terminator
        }

        private void AddByte(byte b)
        {
            _block[_blockLen++] = b;
            if (_blockLen == 255)
                FlushBlock();
        }

        private void FlushBlock()
        {
            if (_blockLen == 0) return;
            writer.Write((byte)_blockLen);
            writer.Write(_block, 0, _blockLen);
            _blockLen = 0;
        }
    }
}
