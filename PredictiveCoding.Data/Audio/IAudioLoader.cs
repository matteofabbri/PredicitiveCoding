using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Audio;

/// <summary>
/// Decodes audio files of various formats into a tensor of float samples.
/// </summary>
public interface IAudioLoader
{
    /// <summary>
    /// Decodes the audio file at <paramref name="path"/> into an <see cref="AudioClip"/>.
    /// </summary>
    /// <param name="path">Path to a supported audio file (e.g. WAV, MP3, AIFF).</param>
    /// <param name="options">Decoding options; <c>null</c> uses <see cref="AudioLoadOptions.Default"/>.</param>
    AudioClip Load(string path, AudioLoadOptions? options = null);

    /// <summary>
    /// Decodes the audio file at <paramref name="path"/> and returns only the
    /// 1-D <see cref="DenseTensor{T}"/> of samples (metadata is discarded).
    /// </summary>
    DenseTensor<float> LoadSamples(string path, AudioLoadOptions? options = null);
}
