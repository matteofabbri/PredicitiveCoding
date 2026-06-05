using OpenCvSharp;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Video;

/// <summary>
/// Captures frames from a webcam with OpenCV, reusing the same decoding pipeline
/// as the video-file loader. Requires a camera device.
/// </summary>
public static class WebcamCapture
{
    /// <summary>Grabs <paramref name="frameCount"/> frames from a capture device into a <see cref="VideoClip"/>.</summary>
    public static VideoClip CaptureFrames(int frameCount, int deviceIndex = 0, VideoLoadOptions? options = null)
    {
        if (frameCount <= 0) throw new ArgumentOutOfRangeException(nameof(frameCount));
        options ??= VideoLoadOptions.Default;
        OpenCvVideoLoader.ValidateOptions(options);

        using var capture = new VideoCapture(deviceIndex);
        if (!capture.IsOpened())
            throw new InvalidOperationException($"Could not open capture device {deviceIndex}.");

        int channels = options.Grayscale ? 1 : 3;
        var frames = new List<byte[]>(frameCount);
        int width = 0, height = 0;

        using var raw = new Mat();
        for (int i = 0; i < frameCount; i++)
        {
            if (!capture.Read(raw) || raw.Empty())
                break;
            using Mat processed = OpenCvVideoLoader.ProcessFrame(raw, options);
            if (width == 0)
            {
                width = processed.Width;
                height = processed.Height;
            }
            frames.Add(OpenCvVideoLoader.MatToBytes(processed));
        }

        if (frames.Count == 0)
            throw new InvalidDataException($"No frames could be captured from device {deviceIndex}.");

        DenseTensor<float> tensor = OpenCvVideoLoader.AssembleTensor(frames, height, width, channels, options.Normalize);
        return new VideoClip(tensor, frames.Count, height, width, channels, capture.Fps);
    }
}
