using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Images;

/// <summary>
/// A decoded image: the pixels as a <see cref="DenseTensor{T}"/> plus its shape.
/// </summary>
/// <remarks>
/// <see cref="Pixels"/> is row-major <c>[height, width, channels]</c> for colour
/// (channels = 3, RGB order) or <c>[height, width]</c> for grayscale. Values are
/// 0..1 unless normalisation was disabled, in which case they are 0..255.
/// </remarks>
public sealed class ImageData
{
    /// <summary>The decoded pixels.</summary>
    public DenseTensor<float> Pixels { get; }

    /// <summary>Height in pixels.</summary>
    public int Height { get; }

    /// <summary>Width in pixels.</summary>
    public int Width { get; }

    /// <summary>Channels per pixel (3 for RGB, 1 for grayscale).</summary>
    public int Channels { get; }

    public ImageData(DenseTensor<float> pixels, int height, int width, int channels)
    {
        ArgumentNullException.ThrowIfNull(pixels);
        if (height <= 0)
            throw new ArgumentOutOfRangeException(nameof(height), height, "Height must be positive.");
        if (width <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), width, "Width must be positive.");
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), channels, "Channel count must be positive.");

        long expected = (long)height * width * channels;
        if (pixels.Length != expected)
            throw new ArgumentException(
                $"Pixel tensor length {pixels.Length} does not match height*width*channels ({expected}).", nameof(pixels));

        Pixels = pixels;
        Height = height;
        Width = width;
        Channels = channels;
    }
}
