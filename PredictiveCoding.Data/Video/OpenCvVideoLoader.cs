using System.Runtime.InteropServices;
using OpenCvSharp;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Video;

/// <summary>
/// <see cref="IVideoLoader"/> backed by OpenCV (OpenCvSharp). Decodes any format
/// the OpenCV <see cref="VideoCapture"/> backend understands (MP4, AVI, MOV, …)
/// into a tensor of float pixels.
/// </summary>
public sealed class OpenCvVideoLoader : IVideoLoader
{
    /// <summary>Shared, stateless instance.</summary>
    public static OpenCvVideoLoader Instance { get; } = new();

    /// <inheritdoc />
    public VideoClip Load(string path, VideoLoadOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("Video file not found.", path);

        options ??= VideoLoadOptions.Default;
        ValidateOptions(options);

        using var capture = new VideoCapture(path);
        if (!capture.IsOpened())
            throw new InvalidOperationException(
                $"Could not open video '{path}'. The format or codec may be unsupported by the OpenCV backend.");

        double fps = capture.Fps;
        int channels = options.Grayscale ? 1 : 3;
        int stride = options.FrameStride <= 0 ? 1 : options.FrameStride;

        var frames = new List<byte[]>();
        int width = 0, height = 0;
        int sourceIndex = 0;

        using var raw = new Mat();
        while (capture.Read(raw))
        {
            if (raw.Empty())
                break;

            if (sourceIndex++ % stride != 0)
                continue;

            using Mat processed = ProcessFrame(raw, options);
            if (width == 0)
            {
                width = processed.Width;
                height = processed.Height;
            }
            frames.Add(MatToBytes(processed));

            if (options.MaxFrames is int max && frames.Count >= max)
                break;
        }

        if (frames.Count == 0)
            throw new InvalidDataException($"No frames could be decoded from '{path}'.");

        DenseTensor<float> tensor = AssembleTensor(frames, height, width, channels, options.Normalize);
        return new VideoClip(tensor, frames.Count, height, width, channels, fps);
    }

    /// <inheritdoc />
    public DenseTensor<float> LoadFrames(string path, VideoLoadOptions? options = null)
        => Load(path, options).Frames;

    internal static void ValidateOptions(VideoLoadOptions options)
    {
        bool hasWidth = options.TargetWidth is not null;
        bool hasHeight = options.TargetHeight is not null;
        if (hasWidth != hasHeight)
            throw new ArgumentException("TargetWidth and TargetHeight must be set together.", nameof(options));
        if (hasWidth && (options.TargetWidth <= 0 || options.TargetHeight <= 0))
            throw new ArgumentException("Target dimensions must be positive.", nameof(options));
        if (options.MaxFrames is int max && max <= 0)
            throw new ArgumentException("MaxFrames must be positive when set.", nameof(options));
    }

    // Converts a decoded BGR frame to the requested colour space and size.
    internal static Mat ProcessFrame(Mat source, VideoLoadOptions options)
    {
        var result = new Mat();
        // OpenCV decodes to BGR; emit grayscale or RGB.
        Cv2.CvtColor(source, result,
            options.Grayscale ? ColorConversionCodes.BGR2GRAY : ColorConversionCodes.BGR2RGB);

        if (options.TargetWidth is int tw && options.TargetHeight is int th &&
            (result.Width != tw || result.Height != th))
        {
            var resized = new Mat();
            Cv2.Resize(result, resized, new Size(tw, th));
            result.Dispose();
            result = resized;
        }
        return result;
    }

    // Copies a continuous 8-bit Mat into a flat byte buffer (row-major, interleaved).
    internal static byte[] MatToBytes(Mat mat)
    {
        Mat continuous = mat.IsContinuous() ? mat : mat.Clone();
        try
        {
            int length = checked((int)(continuous.Total() * continuous.ElemSize()));
            var buffer = new byte[length];
            Marshal.Copy(continuous.Data, buffer, 0, length);
            return buffer;
        }
        finally
        {
            if (!ReferenceEquals(continuous, mat))
                continuous.Dispose();
        }
    }

    internal static DenseTensor<float> AssembleTensor(
        List<byte[]> frames, int height, int width, int channels, bool normalize)
    {
        int frameCount = frames.Count;
        long perFrame = (long)height * width * channels;
        int total = checked((int)(perFrame * frameCount));

        var data = new float[total];
        float scale = normalize ? 1f / 255f : 1f;

        int offset = 0;
        foreach (byte[] frame in frames)
        {
            for (int i = 0; i < frame.Length; i++)
                data[offset + i] = frame[i] * scale;
            offset += frame.Length;
        }

        int[] dimensions = channels == 1
            ? [frameCount, height, width]
            : [frameCount, height, width, channels];
        return new DenseTensor<float>(data, dimensions);
    }
}
