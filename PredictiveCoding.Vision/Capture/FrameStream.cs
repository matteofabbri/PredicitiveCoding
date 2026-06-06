using System.Runtime.InteropServices;
using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PredictiveCoding.Vision.Capture;

/// <summary>
/// Continuous live frame source built on OpenCV's <see cref="VideoCapture"/>.
/// Accepts a webcam index (<c>"0"</c>), a video-file path, or an <c>rtsp://</c> /
/// <c>http://</c> network-camera URL, and streams decoded RGB frames one at a time
/// — the streaming loop the batch-oriented <c>WebcamCapture</c> does not provide.
/// </summary>
public static class FrameStream
{
    /// <summary>
    /// Opens <paramref name="source"/> and streams frames as <see cref="Image{Rgb24}"/>.
    /// For a webcam or network camera the sequence runs until <paramref name="cancel"/>
    /// is requested or the device stops returning frames; for a file it ends at EOF.
    /// Only every <paramref name="frameStride"/>-th frame is yielded. The caller owns
    /// (and must dispose) each returned image.
    /// </summary>
    public static IEnumerable<Image<Rgb24>> Read(
        string source, int frameStride = 1, CancellationToken cancel = default)
    {
        if (frameStride < 1) frameStride = 1;

        using VideoCapture capture = OpenSource(source);
        if (!capture.IsOpened())
            throw new InvalidOperationException(
                $"Could not open capture source '{source}'. Check the camera index, file path or URL.");

        using var raw = new Mat();
        using var rgb = new Mat();
        long index = 0;
        while (!cancel.IsCancellationRequested && capture.Read(raw) && !raw.Empty())
        {
            if (index++ % frameStride != 0) continue;
            Cv2.CvtColor(raw, rgb, ColorConversionCodes.BGR2RGB);
            yield return ToImage(rgb);
        }
    }

    /// <summary>Grabs a single frame from <paramref name="source"/>, or <c>null</c> if none is available.</summary>
    public static Image<Rgb24>? GrabOne(string source)
    {
        using VideoCapture capture = OpenSource(source);
        if (!capture.IsOpened()) return null;
        using var raw = new Mat();
        if (!capture.Read(raw) || raw.Empty()) return null;
        using var rgb = new Mat();
        Cv2.CvtColor(raw, rgb, ColorConversionCodes.BGR2RGB);
        return ToImage(rgb);
    }

    private static VideoCapture OpenSource(string source)
        => int.TryParse(source, out int deviceIndex)
            ? new VideoCapture(deviceIndex)
            : new VideoCapture(source);

    // Copies a continuous 8-bit, 3-channel RGB Mat into an ImageSharp image.
    private static Image<Rgb24> ToImage(Mat rgb)
    {
        int w = rgb.Width, h = rgb.Height;
        Mat continuous = rgb.IsContinuous() ? rgb : rgb.Clone();
        try
        {
            int length = checked((int)(continuous.Total() * continuous.ElemSize())); // w*h*3
            byte[] buffer = new byte[length];
            Marshal.Copy(continuous.Data, buffer, 0, length);
            return Image.LoadPixelData<Rgb24>(buffer, w, h);
        }
        finally
        {
            if (!ReferenceEquals(continuous, rgb)) continuous.Dispose();
        }
    }
}
