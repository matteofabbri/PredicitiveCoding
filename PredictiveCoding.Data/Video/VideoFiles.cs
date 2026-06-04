using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Video;

/// <summary>
/// Convenience entry point for turning a video file on disk into a tensor of
/// frames. Backed by <see cref="OpenCvVideoLoader"/>.
/// </summary>
public static class VideoFiles
{
    /// <summary>Decodes a video file into a <see cref="VideoClip"/> (frames + metadata).</summary>
    public static VideoClip Load(string path, VideoLoadOptions? options = null)
        => OpenCvVideoLoader.Instance.Load(path, options);

    /// <summary>Decodes a video file and returns just the tensor of frames.</summary>
    public static DenseTensor<float> LoadFrames(string path, VideoLoadOptions? options = null)
        => OpenCvVideoLoader.Instance.LoadFrames(path, options);
}
