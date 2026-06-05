using PredictiveCoding.Core;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PredictiveCoding.Data.Images;

/// <summary>
/// <see cref="IImageLoader"/> backed by SixLabors.ImageSharp (pure managed,
/// cross-platform). Decodes PNG, JPEG, BMP, GIF, WebP, TGA and more.
/// </summary>
public sealed class ImageSharpLoader : IImageLoader
{
    /// <summary>Shared, stateless instance.</summary>
    public static ImageSharpLoader Instance { get; } = new();

    /// <inheritdoc />
    public ImageData Load(string path, ImageLoadOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("Image file not found.", path);

        options ??= ImageLoadOptions.Default;
        ValidateOptions(options);

        using Image<Rgb24> image = Image.Load<Rgb24>(path);

        if (options.TargetWidth is int tw && options.TargetHeight is int th &&
            (image.Width != tw || image.Height != th))
        {
            image.Mutate(c => c.Resize(tw, th));
        }

        int width = image.Width;
        int height = image.Height;
        int channels = options.Grayscale ? 1 : 3;

        byte[] bytes;
        if (options.Grayscale)
        {
            using Image<L8> gray = image.CloneAs<L8>();
            bytes = new byte[width * height];
            gray.CopyPixelDataTo(bytes);
        }
        else
        {
            bytes = new byte[width * height * 3];
            image.CopyPixelDataTo(bytes);
        }

        DenseTensor<float> tensor = ToTensor(bytes, height, width, channels, options.Normalize);
        return new ImageData(tensor, height, width, channels);
    }

    /// <inheritdoc />
    public DenseTensor<float> LoadPixels(string path, ImageLoadOptions? options = null)
        => Load(path, options).Pixels;

    /// <summary>Decodes every frame of an animated image (GIF/APNG/WebP) into a tensor.</summary>
    public AnimatedImage LoadAnimation(string path, ImageLoadOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("Image file not found.", path);

        options ??= ImageLoadOptions.Default;
        ValidateOptions(options);

        using Image<Rgb24> image = Image.Load<Rgb24>(path);
        int frameCount = image.Frames.Count;
        int channels = options.Grayscale ? 1 : 3;
        int width = 0, height = 0;
        var frames = new List<byte[]>(frameCount);

        for (int i = 0; i < frameCount; i++)
        {
            using Image<Rgb24> frame = image.Frames.CloneFrame(i);
            if (options.TargetWidth is int tw && options.TargetHeight is int th &&
                (frame.Width != tw || frame.Height != th))
            {
                frame.Mutate(c => c.Resize(tw, th));
            }

            if (width == 0)
            {
                width = frame.Width;
                height = frame.Height;
            }

            if (options.Grayscale)
            {
                using Image<L8> gray = frame.CloneAs<L8>();
                var bytes = new byte[width * height];
                gray.CopyPixelDataTo(bytes);
                frames.Add(bytes);
            }
            else
            {
                var bytes = new byte[width * height * 3];
                frame.CopyPixelDataTo(bytes);
                frames.Add(bytes);
            }
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
