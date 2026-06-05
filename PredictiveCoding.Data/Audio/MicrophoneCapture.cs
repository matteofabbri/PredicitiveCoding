using NAudio.Wave;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Audio;

/// <summary>
/// Captures live audio from the default input device with NAudio. The recording
/// path needs audio hardware; the PCM-to-samples conversion is static and runs
/// offline.
/// </summary>
public static class MicrophoneCapture
{
    /// <summary>Records mono audio for <paramref name="seconds"/> and returns the samples in [-1, 1].</summary>
    public static async Task<DenseTensor<float>> RecordAsync(
        int seconds, int sampleRate = 16000, CancellationToken cancellationToken = default)
    {
        if (seconds <= 0) throw new ArgumentOutOfRangeException(nameof(seconds));
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));

        var samples = new List<float>(seconds * sampleRate);
        using var waveIn = new WaveInEvent
        {
            WaveFormat = new WaveFormat(sampleRate, 16, 1),
            BufferMilliseconds = 50,
        };

        waveIn.DataAvailable += (_, e) =>
        {
            for (int i = 0; i + 1 < e.BytesRecorded; i += 2)
                samples.Add(BitConverter.ToInt16(e.Buffer, i) / 32768f);
        };

        waveIn.StartRecording();
        try
        {
            await Task.Delay(TimeSpan.FromSeconds(seconds), cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            waveIn.StopRecording();
        }

        return new DenseTensor<float>(samples.Count > 0 ? [.. samples] : [0f], Math.Max(1, samples.Count));
    }

    /// <summary>
    /// Converts interleaved little-endian PCM into mono float samples in [-1, 1]
    /// (channels are averaged). Supports 16- and 8-bit PCM.
    /// </summary>
    public static DenseTensor<float> PcmToSamples(byte[] pcm, int bitsPerSample = 16, int channels = 1)
    {
        ArgumentNullException.ThrowIfNull(pcm);
        if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
        if (bitsPerSample is not (8 or 16)) throw new ArgumentOutOfRangeException(nameof(bitsPerSample), "Only 8- or 16-bit PCM is supported.");

        int bytesPerSample = bitsPerSample / 8;
        int frameBytes = bytesPerSample * channels;
        int frames = frameBytes > 0 ? pcm.Length / frameBytes : 0;
        if (frames == 0)
            throw new InvalidDataException("PCM buffer is too short for a single frame.");

        var samples = new float[frames];
        for (int f = 0; f < frames; f++)
        {
            float sum = 0f;
            for (int c = 0; c < channels; c++)
            {
                int offset = f * frameBytes + c * bytesPerSample;
                sum += bitsPerSample == 16
                    ? BitConverter.ToInt16(pcm, offset) / 32768f
                    : (pcm[offset] - 128) / 128f;
            }
            samples[f] = sum / channels;
        }
        return new DenseTensor<float>(samples, frames);
    }
}
