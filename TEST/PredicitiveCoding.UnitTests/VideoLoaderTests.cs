using OpenCvSharp;
using PredictiveCoding.Core;
using PredictiveCoding.Data.Video;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Round-trip tests for the OpenCV-backed video loader: synthesise an MJPG/AVI
/// clip with OpenCvSharp, decode it back through the loader and assert on shape,
/// channels, size, frame selection and value range. MJPG is lossy, so content
/// checks use loose tolerances.
/// </summary>
public sealed class VideoLoaderTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"pcdata_video_{Guid.NewGuid():N}");

    public VideoLoaderTests() => Directory.CreateDirectory(_tmp);

    public void Dispose()
    {
        if (Directory.Exists(_tmp)) Directory.Delete(_tmp, true);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// <summary>Writes an MJPG/AVI clip whose frames are solid colours (BGR).</summary>
    private string WriteVideo(int frameCount, int width, int height, double fps = 10, Func<int, Scalar>? color = null)
    {
        color ??= i => new Scalar(i * 10 % 256, 100, 200); // B, G, R
        string path = Path.Combine(_tmp, $"{Guid.NewGuid():N}.avi");
        using var writer = new VideoWriter(path, VideoWriter.FourCC("MJPG"), fps, new Size(width, height), isColor: true);
        Assert.True(writer.IsOpened(), "VideoWriter failed to open — MJPG codec unavailable.");
        for (int i = 0; i < frameCount; i++)
        {
            using var frame = new Mat(new Size(width, height), MatType.CV_8UC3, color(i));
            writer.Write(frame);
        }
        return path;
    }

    private static float ChannelMean(VideoClip clip, int channel)
    {
        float[] buffer = clip.Frames.Buffer;
        int channels = clip.Channels;
        double sum = 0;
        long count = 0;
        for (int i = channel; i < buffer.Length; i += channels)
        {
            sum += buffer[i];
            count++;
        }
        return (float)(sum / count);
    }

    // ── Tests ────────────────────────────────────────────────────────────────

    [Fact]
    public void Load_Color_ReturnsExpectedShapeAndRange()
    {
        string path = WriteVideo(frameCount: 10, width: 32, height: 24);

        VideoClip clip = VideoFiles.Load(path);

        Assert.Equal(10, clip.FrameCount);
        Assert.Equal(24, clip.Height);
        Assert.Equal(32, clip.Width);
        Assert.Equal(3, clip.Channels);
        Assert.Equal(4, clip.Frames.Rank);
        Assert.Equal(new[] { 10, 24, 32, 3 }, clip.Frames.Dimensions);
        Assert.Equal(10 * 24 * 32 * 3, clip.Frames.Length);
        Assert.InRange(clip.Fps, 9.0, 11.0);
        Assert.All(clip.Frames.Buffer, v => Assert.InRange(v, 0f, 1.0001f)); // Normalize default
    }

    [Fact]
    public void Load_Grayscale_ReturnsRank3SingleChannel()
    {
        string path = WriteVideo(frameCount: 6, width: 16, height: 12);

        VideoClip clip = VideoFiles.Load(path, new VideoLoadOptions { Grayscale = true });

        Assert.Equal(1, clip.Channels);
        Assert.Equal(3, clip.Frames.Rank);
        Assert.Equal(new[] { 6, 12, 16 }, clip.Frames.Dimensions);
    }

    [Fact]
    public void Load_Resize_ChangesFrameDimensions()
    {
        string path = WriteVideo(frameCount: 4, width: 64, height: 48);

        VideoClip clip = VideoFiles.Load(path, new VideoLoadOptions { TargetWidth = 16, TargetHeight = 12 });

        Assert.Equal(16, clip.Width);
        Assert.Equal(12, clip.Height);
        Assert.Equal(new[] { 4, 12, 16, 3 }, clip.Frames.Dimensions);
    }

    [Fact]
    public void Load_MaxFrames_CapsFrameCount()
    {
        string path = WriteVideo(frameCount: 10, width: 16, height: 16);

        VideoClip clip = VideoFiles.Load(path, new VideoLoadOptions { MaxFrames = 3 });

        Assert.Equal(3, clip.FrameCount);
    }

    [Fact]
    public void Load_FrameStride_SubsamplesFrames()
    {
        string path = WriteVideo(frameCount: 10, width: 16, height: 16);

        VideoClip clip = VideoFiles.Load(path, new VideoLoadOptions { FrameStride = 2 });

        Assert.Equal(5, clip.FrameCount); // frames 0,2,4,6,8
    }

    [Fact]
    public void Load_WithoutNormalize_ReturnsByteRange()
    {
        string path = WriteVideo(frameCount: 3, width: 16, height: 16);

        VideoClip clip = VideoFiles.Load(path, new VideoLoadOptions { Normalize = false });

        float max = 0f;
        foreach (float v in clip.Frames.Buffer) max = MathF.Max(max, v);
        Assert.True(max > 1.5f, $"expected unnormalised pixels, max was {max}");
        Assert.All(clip.Frames.Buffer, v => Assert.InRange(v, 0f, 255.01f));
    }

    [Fact]
    public void Load_RgbOrder_RedFrameIsRedDominant()
    {
        // Pure red in BGR; after BGR->RGB the loader must put red in channel 0.
        string path = WriteVideo(frameCount: 5, width: 16, height: 16, color: _ => new Scalar(0, 0, 255));

        VideoClip clip = VideoFiles.Load(path);

        float red = ChannelMean(clip, 0);
        float blue = ChannelMean(clip, 2);
        Assert.True(red > blue + 0.2f, $"expected red-dominant frame: red={red}, blue={blue}");
    }

    [Fact]
    public void LoadFrames_ReturnsFrameTensor()
    {
        string path = WriteVideo(frameCount: 7, width: 16, height: 16);

        DenseTensor<float> frames = VideoFiles.LoadFrames(path);

        Assert.Equal(4, frames.Rank);
        Assert.Equal(7 * 16 * 16 * 3, frames.Length);
    }

    [Fact]
    public void Load_MissingFile_Throws()
    {
        string path = Path.Combine(_tmp, "does_not_exist.mp4");
        Assert.Throws<FileNotFoundException>(() => VideoFiles.Load(path));
    }

    [Fact]
    public void Load_EmptyOrNullPath_Throws()
    {
        Assert.Throws<ArgumentException>(() => VideoFiles.Load(""));
        Assert.Throws<ArgumentNullException>(() => VideoFiles.Load(null!));
    }

    [Fact]
    public void Load_MismatchedTargetDimensions_Throws()
    {
        string path = WriteVideo(frameCount: 2, width: 16, height: 16);
        Assert.Throws<ArgumentException>(() => VideoFiles.Load(path, new VideoLoadOptions { TargetWidth = 8 }));
    }
}
