using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Audio;

/// <summary>
/// Convenience entry point for the most common task: turning an audio file on
/// disk into a tensor of samples. Backed by <see cref="NAudioLoader"/>.
/// </summary>
public static class AudioFiles
{
    /// <summary>Decodes an audio file into an <see cref="AudioClip"/> (samples + metadata).</summary>
    public static AudioClip Load(string path, AudioLoadOptions? options = null)
        => NAudioLoader.Instance.Load(path, options);

    /// <summary>Decodes an audio file and returns just the 1-D tensor of samples.</summary>
    public static DenseTensor<float> LoadSamples(string path, AudioLoadOptions? options = null)
        => NAudioLoader.Instance.LoadSamples(path, options);
}
