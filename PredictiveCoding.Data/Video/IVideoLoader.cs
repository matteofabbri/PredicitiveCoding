using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Video;

/// <summary>
/// Decodes video files into a tensor of frames.
/// </summary>
public interface IVideoLoader
{
    /// <summary>
    /// Decodes the video file at <paramref name="path"/> into a <see cref="VideoClip"/>.
    /// </summary>
    /// <param name="path">Path to a supported video file (e.g. MP4, AVI, MOV).</param>
    /// <param name="options">Decoding options; <c>null</c> uses <see cref="VideoLoadOptions.Default"/>.</param>
    VideoClip Load(string path, VideoLoadOptions? options = null);

    /// <summary>
    /// Decodes the video file at <paramref name="path"/> and returns only the
    /// <see cref="DenseTensor{T}"/> of frames (metadata is discarded).
    /// </summary>
    DenseTensor<float> LoadFrames(string path, VideoLoadOptions? options = null);
}
