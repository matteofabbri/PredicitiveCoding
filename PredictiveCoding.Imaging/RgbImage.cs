using System.Drawing;
using System.Runtime.InteropServices;
using StbImageSharp;
using StbImageWriteSharp;

namespace PredictiveCoding.Imaging;

/// <summary>
/// In-house 24-bit RGB image: a flat row-major byte buffer (3 bytes per pixel)
/// plus the resize / crop / grayscale operations the vision stack needs.
/// Decoding goes through StbImageSharp (pure C#, public domain) with an in-house
/// Netpbm (.pgm/.ppm) fallback; encoding through StbImageWriteSharp and the
/// in-house <see cref="GifEncoder"/>. No native binaries, no licence gate.
/// <para>
/// <see cref="IDisposable"/> is implemented as a no-op purely so streaming call
/// sites can treat frames uniformly (the buffer is ordinary managed memory).
/// </para>
/// </summary>
public sealed class RgbImage : IDisposable
{
    /// <summary>Image width in pixels.</summary>
    public int Width { get; }

    /// <summary>Image height in pixels.</summary>
    public int Height { get; }

    /// <summary>Raw RGB pixels, row-major, 3 bytes per pixel (R, G, B).</summary>
    public byte[] Pixels { get; }

    public RgbImage(int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        Width = width;
        Height = height;
        Pixels = new byte[checked(width * height * 3)];
    }

    public RgbImage(int width, int height, Rgb24 fill)
        : this(width, height)
    {
        Span<Rgb24> all = MemoryMarshal.Cast<byte, Rgb24>(Pixels.AsSpan());
        all.Fill(fill);
    }

    private RgbImage(int width, int height, byte[] pixels)
    {
        Width = width;
        Height = height;
        Pixels = pixels;
    }

    /// <summary>
    /// Wraps an existing RGB24 buffer (row-major, 3 bytes per pixel) without
    /// copying; the image takes ownership of <paramref name="rgb"/>.
    /// </summary>
    public static RgbImage FromPixelData(byte[] rgb, int width, int height)
    {
        ArgumentNullException.ThrowIfNull(rgb);
        if (rgb.Length != checked(width * height * 3))
            throw new ArgumentException(
                $"Buffer length {rgb.Length} does not match {width}×{height}×3.", nameof(rgb));
        return new RgbImage(width, height, rgb);
    }

    public Rgb24 this[int x, int y]
    {
        get
        {
            int i = (y * Width + x) * 3;
            return new Rgb24(Pixels[i], Pixels[i + 1], Pixels[i + 2]);
        }
        set
        {
            int i = (y * Width + x) * 3;
            Pixels[i] = value.R;
            Pixels[i + 1] = value.G;
            Pixels[i + 2] = value.B;
        }
    }

    /// <summary>Full-image rectangle, for intersection tests.</summary>
    public Rectangle Bounds => new(0, 0, Width, Height);

    /// <summary>Row <paramref name="y"/> reinterpreted as pixels (no copy).</summary>
    public Span<Rgb24> GetRowSpan(int y)
        => MemoryMarshal.Cast<byte, Rgb24>(Pixels.AsSpan(y * Width * 3, Width * 3));

    public RgbImage Clone()
    {
        var copy = new byte[Pixels.Length];
        Pixels.CopyTo(copy, 0);
        return new RgbImage(Width, Height, copy);
    }

    /// <summary>Returns the sub-image at <paramref name="rect"/> (must lie inside the image).</summary>
    public RgbImage Crop(Rectangle rect)
    {
        if (rect.Width < 1 || rect.Height < 1 ||
            rect.X < 0 || rect.Y < 0 || rect.Right > Width || rect.Bottom > Height)
            throw new ArgumentOutOfRangeException(nameof(rect),
                $"Crop {rect} outside {Width}×{Height} image.");

        var result = new RgbImage(rect.Width, rect.Height);
        for (int y = 0; y < rect.Height; y++)
        {
            int src = ((rect.Y + y) * Width + rect.X) * 3;
            Array.Copy(Pixels, src, result.Pixels, y * rect.Width * 3, rect.Width * 3);
        }
        return result;
    }

    /// <summary>
    /// Bilinear resize with half-pixel-centre sampling (the OpenCV
    /// <c>INTER_LINEAR</c> convention). Returns a new image.
    /// </summary>
    public RgbImage Resize(int width, int height)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(width, 1);
        ArgumentOutOfRangeException.ThrowIfLessThan(height, 1);
        if (width == Width && height == Height) return Clone();

        var result = new RgbImage(width, height);
        float sx = (float)Width / width;
        float sy = (float)Height / height;

        for (int dy = 0; dy < height; dy++)
        {
            float srcY = (dy + 0.5f) * sy - 0.5f;
            int y0 = (int)MathF.Floor(srcY);
            float fy = srcY - y0;
            int ya = Math.Clamp(y0, 0, Height - 1);
            int yb = Math.Clamp(y0 + 1, 0, Height - 1);

            int rowA = ya * Width * 3;
            int rowB = yb * Width * 3;
            int dst = dy * width * 3;

            for (int dx = 0; dx < width; dx++)
            {
                float srcX = (dx + 0.5f) * sx - 0.5f;
                int x0 = (int)MathF.Floor(srcX);
                float fx = srcX - x0;
                int xa = Math.Clamp(x0, 0, Width - 1) * 3;
                int xb = Math.Clamp(x0 + 1, 0, Width - 1) * 3;

                float w00 = (1f - fx) * (1f - fy);
                float w10 = fx * (1f - fy);
                float w01 = (1f - fx) * fy;
                float w11 = fx * fy;

                int o = dst + dx * 3;
                for (int c = 0; c < 3; c++)
                {
                    float v = Pixels[rowA + xa + c] * w00 + Pixels[rowA + xb + c] * w10 +
                              Pixels[rowB + xa + c] * w01 + Pixels[rowB + xb + c] * w11;
                    result.Pixels[o + c] = (byte)Math.Clamp((int)MathF.Round(v), 0, 255);
                }
            }
        }
        return result;
    }

    /// <summary>
    /// Converts to 8-bit grayscale using ITU-R BT.709 luminance (the same
    /// coefficients ImageSharp used for <c>L8</c>, keeping descriptors stable).
    /// </summary>
    public byte[] ToGray()
    {
        int n = Width * Height;
        var gray = new byte[n];
        for (int i = 0; i < n; i++)
        {
            int p = i * 3;
            float v = 0.2126f * Pixels[p] + 0.7152f * Pixels[p + 1] + 0.0722f * Pixels[p + 2];
            gray[i] = (byte)Math.Clamp((int)MathF.Round(v), 0, 255);
        }
        return gray;
    }

    // ── Decoding ────────────────────────────────────────────────────────────────

    /// <summary>Decodes PNG, JPEG, BMP, GIF (first frame), TGA, PSD or Netpbm (.pgm/.ppm).</summary>
    public static RgbImage Load(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("Image file not found.", path);
        return Load(File.ReadAllBytes(path));
    }

    /// <inheritdoc cref="Load(string)"/>
    public static RgbImage Load(Stream stream)
    {
        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        return Load(ms.ToArray());
    }

    /// <inheritdoc cref="Load(string)"/>
    public static RgbImage Load(byte[] data)
    {
        ArgumentNullException.ThrowIfNull(data);
        if (NetpbmDecoder.IsNetpbm(data))
            return NetpbmDecoder.Decode(data);

        ImageResult decoded = ImageResult.FromMemory(data, StbImageSharp.ColorComponents.RedGreenBlue);
        return FromPixelData(decoded.Data, decoded.Width, decoded.Height);
    }

    /// <summary>
    /// Decodes every frame of an animated GIF (a non-GIF image yields one frame).
    /// Frames are returned fully composed, as displayed.
    /// </summary>
    public static List<RgbImage> LoadFrames(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("Image file not found.", path);

        byte[] data = File.ReadAllBytes(path);
        bool isGif = data.Length > 3 && data[0] == 'G' && data[1] == 'I' && data[2] == 'F';
        if (!isGif)
            return [Load(data)];

        var frames = new List<RgbImage>();
        using var ms = new MemoryStream(data);
        foreach (AnimatedFrameResult frame in
                 ImageResult.AnimatedGifFramesFromStream(ms, StbImageSharp.ColorComponents.RedGreenBlue))
        {
            var pixels = new byte[frame.Data.Length];
            frame.Data.CopyTo(pixels, 0);   // Stb reuses the frame buffer while iterating
            frames.Add(FromPixelData(pixels, frame.Width, frame.Height));
        }
        return frames.Count > 0 ? frames : [Load(data)];
    }

    // ── Encoding ────────────────────────────────────────────────────────────────

    public void SaveAsPng(string path)
    {
        using FileStream fs = File.Create(path);
        new ImageWriter().WritePng(Pixels, Width, Height,
            StbImageWriteSharp.ColorComponents.RedGreenBlue, fs);
    }

    public void SaveAsJpeg(string path, int quality = 90)
    {
        using FileStream fs = File.Create(path);
        new ImageWriter().WriteJpg(Pixels, Width, Height,
            StbImageWriteSharp.ColorComponents.RedGreenBlue, fs, quality);
    }

    public void SaveAsBmp(string path)
    {
        using FileStream fs = File.Create(path);
        new ImageWriter().WriteBmp(Pixels, Width, Height,
            StbImageWriteSharp.ColorComponents.RedGreenBlue, fs);
    }

    /// <summary>Writes an animated GIF via the in-house <see cref="GifEncoder"/>.</summary>
    public static void SaveAsGif(string path, IReadOnlyList<RgbImage> frames, int delayCentiseconds = 10)
        => GifEncoder.Save(path, frames, delayCentiseconds);

    /// <summary>No-op; kept so frame consumers can dispose uniformly.</summary>
    public void Dispose()
    {
    }
}
