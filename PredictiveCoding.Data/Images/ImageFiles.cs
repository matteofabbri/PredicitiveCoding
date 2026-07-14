using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Images;

/// <summary>
/// Convenience entry point for turning an image file into a tensor of pixels.
/// Backed by <see cref="RgbImageLoader"/>.
/// </summary>
public static class ImageFiles
{
    /// <summary>Decodes an image into an <see cref="ImageData"/> (pixels + shape).</summary>
    public static ImageData Load(string path, ImageLoadOptions? options = null)
        => RgbImageLoader.Instance.Load(path, options);

    /// <summary>Decodes an image and returns just the pixel tensor.</summary>
    public static DenseTensor<float> LoadPixels(string path, ImageLoadOptions? options = null)
        => RgbImageLoader.Instance.LoadPixels(path, options);

    /// <summary>Decodes every frame of an animated image into a tensor.</summary>
    public static AnimatedImage LoadAnimation(string path, ImageLoadOptions? options = null)
        => RgbImageLoader.Instance.LoadAnimation(path, options);
}
