namespace PredictiveCoding.Data.Images;

/// <summary>
/// Options controlling how an image file is decoded into a tensor.
/// </summary>
public sealed class ImageLoadOptions
{
    /// <summary>
    /// When <c>true</c> the image is converted to a single grayscale channel and
    /// the tensor has shape <c>[height, width]</c>. When <c>false</c> (default) it
    /// is RGB with shape <c>[height, width, 3]</c>.
    /// </summary>
    public bool Grayscale { get; init; }

    /// <summary>Target width in pixels. Must be set together with <see cref="TargetHeight"/>.</summary>
    public int? TargetWidth { get; init; }

    /// <summary>Target height in pixels. Must be set together with <see cref="TargetWidth"/>.</summary>
    public int? TargetHeight { get; init; }

    /// <summary>If <c>true</c> (default) pixel values are scaled from 0..255 to 0..1.</summary>
    public bool Normalize { get; init; } = true;

    /// <summary>Shared instance with the default settings (RGB, native size, normalised).</summary>
    public static ImageLoadOptions Default { get; } = new();
}
