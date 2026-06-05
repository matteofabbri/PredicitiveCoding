using PredictiveCoding.Core;
using PredictiveCoding.Data.Audio;

namespace PredicitiveCoding.UnitTests;

public sealed class SpectrogramTests
{
    private static DenseTensor<float> Sine(int samples, int sampleRate, float freq)
    {
        var data = new float[samples];
        for (int i = 0; i < samples; i++)
            data[i] = MathF.Sin(2f * MathF.PI * freq * i / sampleRate);
        return new DenseTensor<float>(data, samples);
    }

    [Fact]
    public void Magnitude_ReturnsTimeByFrequency()
    {
        DenseTensor<float> signal = Sine(8000, 16000, 440f);

        DenseTensor<float> spectrogram = Spectrogram.Magnitude(signal, frameSize: 1024, hopSize: 256);

        Assert.Equal(2, spectrogram.Rank);
        Assert.True(spectrogram.Dimensions[0] > 0); // frames
        Assert.True(spectrogram.Dimensions[1] > 0); // frequency bins
        Assert.All(spectrogram.Buffer, v => Assert.True(float.IsFinite(v)));
    }

    [Fact]
    public void Mfcc_ReturnsRequestedFeatureCount()
    {
        DenseTensor<float> signal = Sine(8000, 16000, 440f);

        DenseTensor<float> mfcc = Spectrogram.Mfcc(signal, sampleRate: 16000, featureCount: 13);

        Assert.Equal(2, mfcc.Rank);
        Assert.True(mfcc.Dimensions[0] > 0);
        Assert.Equal(13, mfcc.Dimensions[1]);
    }

    [Fact]
    public void Magnitude_RejectsInvalidSizes()
    {
        DenseTensor<float> signal = Sine(2048, 16000, 200f);
        Assert.Throws<ArgumentOutOfRangeException>(() => Spectrogram.Magnitude(signal, frameSize: 0));
    }
}
