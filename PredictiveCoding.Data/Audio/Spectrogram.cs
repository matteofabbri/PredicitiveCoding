using NWaves.FeatureExtractors;
using NWaves.FeatureExtractors.Options;
using NWaves.Signals;
using NWaves.Transforms;
using NWaves.Windows;
using PredictiveCoding.Core;

namespace PredictiveCoding.Data.Audio;

/// <summary>
/// Time-frequency features from an audio signal using NWaves: a magnitude
/// spectrogram (STFT) or MFCCs. Both return a <c>[frames, bins]</c> tensor.
/// </summary>
public static class Spectrogram
{
    /// <summary>Computes the magnitude spectrogram (STFT) of a mono signal.</summary>
    public static DenseTensor<float> Magnitude(
        DenseTensor<float> samples, int frameSize = 1024, int hopSize = 256)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (frameSize <= 0) throw new ArgumentOutOfRangeException(nameof(frameSize));
        if (hopSize <= 0) throw new ArgumentOutOfRangeException(nameof(hopSize));

        var stft = new Stft(frameSize, hopSize, WindowType.Hann, frameSize);
        List<float[]> frames = stft.Spectrogram(samples.Buffer, normalize: false);
        return Stack(frames);
    }

    /// <summary>Computes MFCC features of a mono signal.</summary>
    public static DenseTensor<float> Mfcc(
        DenseTensor<float> samples, int sampleRate, int featureCount = 13,
        double frameDuration = 0.025, double hopDuration = 0.010)
    {
        ArgumentNullException.ThrowIfNull(samples);
        if (sampleRate <= 0) throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (featureCount <= 0) throw new ArgumentOutOfRangeException(nameof(featureCount));

        var options = new MfccOptions
        {
            SamplingRate = sampleRate,
            FeatureCount = featureCount,
            FrameDuration = frameDuration,
            HopDuration = hopDuration,
        };
        var extractor = new MfccExtractor(options);
        var signal = new DiscreteSignal(sampleRate, samples.Buffer, allocateNew: true);
        List<float[]> features = extractor.ComputeFrom(signal);
        return Stack(features);
    }

    private static DenseTensor<float> Stack(List<float[]> frames)
    {
        if (frames.Count == 0 || frames[0].Length == 0)
            throw new InvalidDataException("The signal produced no spectral frames (it may be too short).");

        int frameCount = frames.Count;
        int bins = frames[0].Length;
        var data = new float[frameCount * bins];
        for (int i = 0; i < frameCount; i++)
            Array.Copy(frames[i], 0, data, i * bins, Math.Min(bins, frames[i].Length));
        return new DenseTensor<float>(data, frameCount, bins);
    }
}
