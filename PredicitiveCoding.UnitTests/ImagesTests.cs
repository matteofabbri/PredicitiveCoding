using PredictiveCoding.Core;
using PredictiveCoding.Data.Images;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Round-trip tests for the ImageSharp loader: write a PNG (lossless) and decode
/// it back, asserting shape, channels, value range and RGB ordering. Offline.
/// </summary>
public sealed class ImagesTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"pcdata_img_{Guid.NewGuid():N}");

    public ImagesTests() => Directory.CreateDirectory(_tmp);

    public void Dispose()
    {
        if (Directory.Exists(_tmp)) Directory.Delete(_tmp, true);
    }

    private string WritePng(int width, int height, Rgb24 color)
    {
        string path = Path.Combine(_tmp, $"{Guid.NewGuid():N}.png");
        using var image = new Image<Rgb24>(width, height, color);
        image.SaveAsPng(path);
        return path;
    }

    private static float ChannelMean(ImageData image, int channel)
    {
        float[] buffer = image.Pixels.Buffer;
        int channels = image.Channels;
        double sum = 0;
        long count = 0;
        for (int i = channel; i < buffer.Length; i += channels)
        {
            sum += buffer[i];
            count++;
        }
        return (float)(sum / count);
    }

    [Fact]
    public void Load_Color_ReturnsExpectedShapeAndRange()
    {
        string path = WritePng(8, 6, new Rgb24(10, 20, 30));

        ImageData image = ImageFiles.Load(path);

        Assert.Equal(6, image.Height);
        Assert.Equal(8, image.Width);
        Assert.Equal(3, image.Channels);
        Assert.Equal(3, image.Pixels.Rank);
        Assert.Equal(new[] { 6, 8, 3 }, image.Pixels.Dimensions);
        Assert.All(image.Pixels.Buffer, v => Assert.InRange(v, 0f, 1.0001f));
    }

    [Fact]
    public void Load_Grayscale_ReturnsRank2()
    {
        string path = WritePng(8, 6, new Rgb24(128, 128, 128));

        ImageData image = ImageFiles.Load(path, new ImageLoadOptions { Grayscale = true });

        Assert.Equal(1, image.Channels);
        Assert.Equal(2, image.Pixels.Rank);
        Assert.Equal(new[] { 6, 8 }, image.Pixels.Dimensions);
    }

    [Fact]
    public void Load_Resize_ChangesDimensions()
    {
        string path = WritePng(16, 12, new Rgb24(0, 0, 0));

        ImageData image = ImageFiles.Load(path, new ImageLoadOptions { TargetWidth = 4, TargetHeight = 3 });

        Assert.Equal(4, image.Width);
        Assert.Equal(3, image.Height);
    }

    [Fact]
    public void Load_RgbOrder_RedImageIsRedDominant()
    {
        string path = WritePng(8, 8, new Rgb24(255, 0, 0));

        ImageData image = ImageFiles.Load(path);

        Assert.True(ChannelMean(image, 0) > 0.9f); // red channel
        Assert.True(ChannelMean(image, 2) < 0.1f); // blue channel
    }

    [Fact]
    public void Load_WithoutNormalize_ReturnsByteRange()
    {
        string path = WritePng(4, 4, new Rgb24(200, 200, 200));

        ImageData image = ImageFiles.Load(path, new ImageLoadOptions { Normalize = false });

        Assert.All(image.Pixels.Buffer, v => Assert.InRange(v, 199f, 201f));
    }

    [Fact]
    public void LoadPixels_ReturnsTensor()
    {
        string path = WritePng(5, 5, new Rgb24(1, 2, 3));
        DenseTensor<float> pixels = ImageFiles.LoadPixels(path);
        Assert.Equal(5 * 5 * 3, pixels.Length);
    }

    [Fact]
    public void Load_MissingFile_Throws()
        => Assert.Throws<FileNotFoundException>(() => ImageFiles.Load(Path.Combine(_tmp, "nope.png")));

    [Fact]
    public void Load_MismatchedTargetDimensions_Throws()
    {
        string path = WritePng(8, 8, new Rgb24(0, 0, 0));
        Assert.Throws<ArgumentException>(() => ImageFiles.Load(path, new ImageLoadOptions { TargetWidth = 4 }));
    }
}
