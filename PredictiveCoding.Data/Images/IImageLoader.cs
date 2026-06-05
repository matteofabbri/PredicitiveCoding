using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Images;

/// <summary>
/// Decodes image files of various formats into a tensor of pixels.
/// </summary>
public interface IImageLoader
{
    /// <summary>Decodes the image at <paramref name="path"/> into an <see cref="ImageData"/>.</summary>
    ImageData Load(string path, ImageLoadOptions? options = null);

    /// <summary>Decodes the image at <paramref name="path"/> and returns only the pixel tensor.</summary>
    DenseTensor<float> LoadPixels(string path, ImageLoadOptions? options = null);
}
