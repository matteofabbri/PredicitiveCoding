using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Images;

/// <summary>
/// A decoded animated image (e.g. GIF/APNG): the frames as a tensor plus shape.
/// </summary>
/// <remarks>
/// <see cref="Frames"/> is <c>[frames, height, width, channels]</c> (RGB) or
/// <c>[frames, height, width]</c> (grayscale).
/// </remarks>
public sealed class AnimatedImage
{
    /// <summary>The decoded frames.</summary>
    public DenseTensor<float> Frames { get; }

    /// <summary>Number of frames.</summary>
    public int FrameCount { get; }

    /// <summary>Frame height in pixels.</summary>
    public int Height { get; }

    /// <summary>Frame width in pixels.</summary>
    public int Width { get; }

    /// <summary>Channels per pixel (3 for RGB, 1 for grayscale).</summary>
    public int Channels { get; }

    public AnimatedImage(DenseTensor<float> frames, int frameCount, int height, int width, int channels)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frameCount <= 0 || height <= 0 || width <= 0 || channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(frames), "Frame dimensions must be positive.");

        long expected = (long)frameCount * height * width * channels;
        if (frames.Length != expected)
            throw new ArgumentException(
                $"Frame tensor length {frames.Length} does not match frames*height*width*channels ({expected}).",
                nameof(frames));

        Frames = frames;
        FrameCount = frameCount;
        Height = height;
        Width = width;
        Channels = channels;
    }
}
