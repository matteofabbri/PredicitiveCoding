namespace PredictiveCoding.Data.Audio;

/// <summary>
/// Options controlling how an audio file is decoded into a tensor of samples.
/// </summary>
public sealed class AudioLoadOptions
{
    /// <summary>
    /// When <c>true</c> (default) multi-channel audio is mixed down to a single
    /// channel by averaging, yielding a 1-D signal vector. When <c>false</c> the
    /// samples are returned interleaved with <see cref="AudioClip.Channels"/> set
    /// to the original channel count.
    /// </summary>
    public bool Mono { get; init; } = true;

    /// <summary>
    /// If set, the signal is resampled to this rate (Hz) before being returned.
    /// <c>null</c> keeps the file's native sample rate.
    /// </summary>
    public int? TargetSampleRate { get; init; }

    /// <summary>
    /// If <c>true</c> the samples are peak-normalised so the largest absolute
    /// amplitude becomes 1.0 (a silent signal is left untouched).
    /// </summary>
    public bool Normalize { get; init; }

    /// <summary>Shared instance with the default settings (mono, native rate, no normalisation).</summary>
    public static AudioLoadOptions Default { get; } = new();
}
