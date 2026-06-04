using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Audio;

/// <summary>
/// A decoded audio signal: the raw float samples as a <see cref="DenseTensor{T}"/>
/// plus the metadata needed to interpret them.
/// </summary>
/// <remarks>
/// <see cref="Samples"/> is a 1-D tensor. For mono audio it holds one value per
/// frame; for multi-channel audio the values are interleaved
/// (<c>[f0c0, f0c1, f1c0, f1c1, …]</c>) and <see cref="Channels"/> is &gt; 1.
/// Sample amplitudes follow NAudio's convention, normalised to roughly [-1, 1].
/// </remarks>
public sealed class AudioClip
{
    /// <summary>The decoded samples (interleaved when <see cref="Channels"/> &gt; 1).</summary>
    public DenseTensor<float> Samples { get; }

    /// <summary>Sample rate in Hz.</summary>
    public int SampleRate { get; }

    /// <summary>Number of interleaved channels (1 for mono).</summary>
    public int Channels { get; }

    /// <summary>Number of frames (samples per channel).</summary>
    public int FrameCount => Channels > 0 ? Samples.Length / Channels : Samples.Length;

    /// <summary>Duration of the clip in seconds.</summary>
    public double DurationSeconds => SampleRate > 0 ? (double)FrameCount / SampleRate : 0.0;

    public AudioClip(DenseTensor<float> samples, int sampleRate, int channels)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate), sampleRate, "Sample rate must be positive.");
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels), channels, "Channel count must be positive.");
        if (samples.Rank != 1)
            throw new ArgumentException("Samples must be a 1-D tensor.", nameof(samples));
        if (samples.Length % channels != 0)
            throw new ArgumentException(
                $"Sample count {samples.Length} is not a multiple of the channel count {channels}.", nameof(samples));

        Samples = samples;
        SampleRate = sampleRate;
        Channels = channels;
    }
}
