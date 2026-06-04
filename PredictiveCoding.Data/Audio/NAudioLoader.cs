using NAudio.Wave;
using NAudio.Wave.SampleProviders;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Audio;

/// <summary>
/// <see cref="IAudioLoader"/> backed by NAudio. Handles every format NAudio's
/// <see cref="AudioFileReader"/> understands (WAV, MP3, AIFF, and — on Windows —
/// any format Media Foundation exposes), returning the samples as 32-bit floats.
/// </summary>
public sealed class NAudioLoader : IAudioLoader
{
    /// <summary>Shared, stateless instance.</summary>
    public static NAudioLoader Instance { get; } = new();

    /// <inheritdoc />
    public AudioClip Load(string path, AudioLoadOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (!File.Exists(path))
            throw new FileNotFoundException("Audio file not found.", path);

        using var reader = new AudioFileReader(path);
        return Load(reader, options);
    }

    /// <inheritdoc />
    public DenseTensor<float> LoadSamples(string path, AudioLoadOptions? options = null)
        => Load(path, options).Samples;

    /// <summary>
    /// Decodes any NAudio <see cref="ISampleProvider"/> (e.g. an
    /// <see cref="AudioFileReader"/> or a custom source) into an <see cref="AudioClip"/>.
    /// </summary>
    public AudioClip Load(ISampleProvider source, AudioLoadOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(source);
        options ??= AudioLoadOptions.Default;

        int channels = Math.Max(1, source.WaveFormat.Channels);
        int sampleRate = source.WaveFormat.SampleRate;

        // Resampling preserves the channel layout, so apply it before mixing down.
        if (options.TargetSampleRate is int target && target > 0 && target != sampleRate)
        {
            source = new WdlResamplingSampleProvider(source, target);
            sampleRate = target;
        }

        float[] interleaved = ReadAllSamples(source);

        if (options.Mono && channels > 1)
        {
            interleaved = MixToMono(interleaved, channels);
            channels = 1;
        }

        if (options.Normalize)
            PeakNormalize(interleaved);

        if (interleaved.Length == 0)
            throw new InvalidDataException("The audio source produced no samples.");

        var tensor = new DenseTensor<float>(interleaved, interleaved.Length);
        return new AudioClip(tensor, sampleRate, channels);
    }

    /// <summary>Drains a sample provider into a flat interleaved float array.</summary>
    private static float[] ReadAllSamples(ISampleProvider source)
    {
        int channels = Math.Max(1, source.WaveFormat.Channels);
        int rate = Math.Max(1, source.WaveFormat.SampleRate);

        // ~0.5 s read chunk, rounded to a whole number of frames.
        int chunkFrames = Math.Max(1, rate / 2);
        var chunk = new float[chunkFrames * channels];

        var samples = new List<float>(chunk.Length);
        int read;
        while ((read = source.Read(chunk, 0, chunk.Length)) > 0)
        {
            for (int i = 0; i < read; i++)
                samples.Add(chunk[i]);
        }
        return samples.ToArray();
    }

    /// <summary>Averages interleaved channels down to a single mono channel.</summary>
    private static float[] MixToMono(float[] interleaved, int channels)
    {
        if (channels <= 1)
            return interleaved;

        int frames = interleaved.Length / channels;
        var mono = new float[frames];
        for (int f = 0; f < frames; f++)
        {
            int baseIdx = f * channels;
            float sum = 0f;
            for (int c = 0; c < channels; c++)
                sum += interleaved[baseIdx + c];
            mono[f] = sum / channels;
        }
        return mono;
    }

    /// <summary>Scales the signal so the largest absolute amplitude becomes 1.0.</summary>
    private static void PeakNormalize(float[] samples)
    {
        float peak = 0f;
        foreach (float s in samples)
        {
            float a = MathF.Abs(s);
            if (a > peak)
                peak = a;
        }
        if (peak <= 1e-9f)
            return;

        float gain = 1f / peak;
        for (int i = 0; i < samples.Length; i++)
            samples[i] *= gain;
    }
}
