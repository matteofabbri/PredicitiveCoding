using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Video;

/// <summary>
/// A decoded video: the frames as a <see cref="DenseTensor{T}"/> plus the
/// metadata needed to interpret them.
/// </summary>
/// <remarks>
/// <see cref="Frames"/> is laid out row-major as <c>[frames, height, width, channels]</c>
/// for colour video (channels = 3, RGB order) or <c>[frames, height, width]</c> for
/// grayscale (channels = 1). Pixel values are normalised to 0..1 unless
/// <see cref="VideoLoadOptions.Normalize"/> was disabled, in which case they are 0..255.
/// </remarks>
public sealed class VideoClip
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

    /// <summary>Frames per second reported by the container (0 if unknown).</summary>
    public double Fps { get; }

    /// <summary>Duration in seconds derived from <see cref="FrameCount"/> and <see cref="Fps"/>.</summary>
    public double DurationSeconds => Fps > 0 ? FrameCount / Fps : 0.0;

    public VideoClip(DenseTensor<float> frames, int frameCount, int height, int width, int channels, double fps)
    {
        ArgumentNullException.ThrowIfNull(frames);
        if (frameCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(frameCount), frameCount, "Frame count must be positive.");
        if (height <= 0)
            throw new ArgumentOutOfRangeException(nameof(height), height, "Height must be positive.");
        if (width <= 0)
            throw new ArgumentOutOfRangeException(nameof(width), width, "Width must be positive.");
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), channels, "Channel count must be positive.");

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
        Fps = fps;
    }
}
