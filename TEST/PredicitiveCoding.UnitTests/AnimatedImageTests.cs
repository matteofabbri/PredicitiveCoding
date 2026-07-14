using PredictiveCoding.Data.Images;
using PredictiveCoding.Imaging;

namespace PredicitiveCoding.UnitTests;

public sealed class AnimatedImageTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"pcdata_gif_{Guid.NewGuid():N}");

    public AnimatedImageTests() => Directory.CreateDirectory(_tmp);

    public void Dispose()
    {
        if (Directory.Exists(_tmp)) Directory.Delete(_tmp, true);
    }

    private string WriteGif(int width, int height)
    {
        string path = Path.Combine(_tmp, $"{Guid.NewGuid():N}.gif");
        RgbImage[] frames =
        [
            new RgbImage(width, height, new Rgb24(255, 0, 0)),
            new RgbImage(width, height, new Rgb24(0, 0, 255)),
        ];
        RgbImage.SaveAsGif(path, frames);
        return path;
    }

    [Fact]
    public void LoadAnimation_ReadsAllFrames()
    {
        string path = WriteGif(4, 4);

        AnimatedImage animation = ImageFiles.LoadAnimation(path);

        Assert.Equal(2, animation.FrameCount);
        Assert.Equal(4, animation.Width);
        Assert.Equal(4, animation.Height);
        Assert.Equal(3, animation.Channels);
        Assert.Equal(new[] { 2, 4, 4, 3 }, animation.Frames.Dimensions);
    }

    [Fact]
    public void LoadAnimation_Grayscale_DropsChannelAxis()
    {
        string path = WriteGif(4, 4);
        AnimatedImage animation = ImageFiles.LoadAnimation(path, new ImageLoadOptions { Grayscale = true });
        Assert.Equal(1, animation.Channels);
        Assert.Equal(new[] { 2, 4, 4 }, animation.Frames.Dimensions);
    }
}
