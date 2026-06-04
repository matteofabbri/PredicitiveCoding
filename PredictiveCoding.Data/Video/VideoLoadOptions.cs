namespace PredictiveCoding.Data.Video;

/// <summary>
/// Options controlling how a video file is decoded into a tensor of frames.
/// </summary>
public sealed class VideoLoadOptions
{
    /// <summary>
    /// When <c>true</c>, frames are converted to a single grayscale channel and
    /// the resulting tensor has shape <c>[frames, height, width]</c>. When
    /// <c>false</c> (default) frames are RGB and the tensor is
    /// <c>[frames, height, width, 3]</c>.
    /// </summary>
    public bool Grayscale { get; init; }

    /// <summary>Target frame width in pixels. Must be set together with <see cref="TargetHeight"/>.</summary>
    public int? TargetWidth { get; init; }

    /// <summary>Target frame height in pixels. Must be set together with <see cref="TargetWidth"/>.</summary>
    public int? TargetHeight { get; init; }

    /// <summary>
    /// Keep one frame out of every <c>FrameStride</c> (1 = every frame, 2 = every
    /// other frame, …). Values &lt;= 0 are treated as 1.
    /// </summary>
    public int FrameStride { get; init; } = 1;

    /// <summary>Maximum number of frames to return. <c>null</c> decodes the whole video.</summary>
    public int? MaxFrames { get; init; }

    /// <summary>
    /// If <c>true</c> (default) pixel values are scaled from 0..255 to 0..1.
    /// </summary>
    public bool Normalize { get; init; } = true;

    /// <summary>Shared instance with the default settings (RGB, native size, all frames, normalised).</summary>
    public static VideoLoadOptions Default { get; } = new();
}
