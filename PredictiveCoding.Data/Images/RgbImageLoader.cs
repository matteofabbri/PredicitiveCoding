using PredictiveCoding.Core;
using PredictiveCoding.Imaging;

namespace PredictiveCoding.Data.Images;

/// <summary>
/// <see cref="IImageLoader"/> backed by the in-house <see cref="RgbImage"/>
/// (StbImageSharp decode — pure managed, public domain). Decodes PNG, JPEG,
/// BMP, GIF, TGA, PSD and Netpbm (.pgm/.ppm).
/// </summary>
public sealed class RgbImageLoader : IImageLoader
{
    /// <summary>Shared, stateless instance.</summary>
    public static RgbImageLoader Instance { get; } = new();

    /// <inheritdoc />
    public ImageData Load(string path, ImageLoadOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("Image file not found.", path);

        options ??= ImageLoadOptions.Default;
        ValidateOptions(options);

        RgbImage image = RgbImage.Load(path);

        if (options.TargetWidth is int tw && options.TargetHeight is int th &&
            (image.Width != tw || image.Height != th))
        {
            image = image.Resize(tw, th);
        }

        int width = image.Width;
        int height = image.Height;
        int channels = options.Grayscale ? 1 : 3;

        byte[] bytes = options.Grayscale ? image.ToGray() : image.Pixels;

        DenseTensor<float> tensor = ToTensor(bytes, height, width, channels, options.Normalize);
        return new ImageData(tensor, height, width, channels);
    }

    /// <inheritdoc />
    public DenseTensor<float> LoadPixels(string path, ImageLoadOptions? options = null)
        => Load(path, options).Pixels;

    /// <summary>Decodes every frame of an animated image (GIF) into a tensor.</summary>
    public AnimatedImage LoadAnimation(string path, ImageLoadOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("Image file not found.", path);

        options ??= ImageLoadOptions.Default;
        ValidateOptions(options);

        List<RgbImage> decoded = RgbImage.LoadFrames(path);
        int frameCount = decoded.Count;
        int channels = options.Grayscale ? 1 : 3;
        int width = 0, height = 0;
        var frames = new List<byte[]>(frameCount);

        foreach (RgbImage rawFrame in decoded)
        {
            RgbImage frame = rawFrame;
            if (options.TargetWidth is int tw && options.TargetHeight is int th &&
                (frame.Width != tw || frame.Height != th))
            {
                frame = frame.Resize(tw, th);
            }

            if (width == 0)
            {
                width = frame.Width;
                height = frame.Height;
            }

            frames.Add(options.Grayscale ? frame.ToGray() : frame.Pixels);
        }

        long perFrame = (long)height * width * channels;
        var data = new float[checked((int)(perFrame * frameCount))];
        float scale = options.Normalize ? 1f / 255f : 1f;
        int offset = 0;
        foreach (byte[] frame in frames)
        {
            for (int i = 0; i < frame.Length; i++)
                data[offset + i] = frame[i] * scale;
            offset += frame.Length;
        }

        int[] dimensions = channels == 1 ? [frameCount, height, width] : [frameCount, height, width, channels];
        var tensor = new DenseTensor<float>(data, dimensions);
        return new AnimatedImage(tensor, frameCount, height, width, channels);
    }

    private static void ValidateOptions(ImageLoadOptions options)
    {
        bool hasWidth = options.TargetWidth is not null;
        bool hasHeight = options.TargetHeight is not null;
        if (hasWidth != hasHeight)
            throw new ArgumentException("TargetWidth and TargetHeight must be set together.", nameof(options));
        if (hasWidth && (options.TargetWidth <= 0 || options.TargetHeight <= 0))
            throw new ArgumentException("Target dimensions must be positive.", nameof(options));
    }

    private static DenseTensor<float> ToTensor(byte[] bytes, int height, int width, int channels, bool normalize)
    {
        var data = new float[bytes.Length];
        float scale = normalize ? 1f / 255f : 1f;
        for (int i = 0; i < bytes.Length; i++)
            data[i] = bytes[i] * scale;

        int[] dimensions = channels == 1 ? [height, width] : [height, width, channels];
        return new DenseTensor<float>(data, dimensions);
    }
}
