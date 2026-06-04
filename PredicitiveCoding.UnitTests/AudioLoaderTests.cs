using NAudio.Wave;
using PredictiveCoding.Core;
using PredictiveCoding.Data.Audio;

namespace PredicitiveCoding.UnitTests;

/// <summary>
/// Round-trip tests for the NAudio-backed loader: synthesise a WAV with NAudio,
/// decode it back through the loader and assert on shape, rate and amplitudes.
/// WAV is used throughout so the tests are deterministic and cross-platform.
/// </summary>
public sealed class AudioLoaderTests : IDisposable
{
    private readonly string _tmp = Path.Combine(Path.GetTempPath(), $"pcdata_audio_{Guid.NewGuid():N}");

    public AudioLoaderTests() => Directory.CreateDirectory(_tmp);

    public void Dispose()
    {
        if (Directory.Exists(_tmp))
            Directory.Delete(_tmp, true);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    /// <summary>Writes an interleaved float buffer as an IEEE-float WAV file.</summary>
    private string WriteWav(float[] interleaved, int sampleRate, int channels)
    {
        string path = Path.Combine(_tmp, $"{Guid.NewGuid():N}.wav");
        var format = WaveFormat.CreateIeeeFloatWaveFormat(sampleRate, channels);
        using (var writer = new WaveFileWriter(path, format))
        {
            writer.WriteSamples(interleaved, 0, interleaved.Length);
        }
        return path;
    }

    private static float[] Sine(int frames, int sampleRate, float freq, float amplitude = 1f)
    {
        var data = new float[frames];
        for (int i = 0; i < frames; i++)
            data[i] = amplitude * MathF.Sin(2f * MathF.PI * freq * i / sampleRate);
        return data;
    }

    // ── Tests ────────────────────────────────────────────────────────────────

    [Fact]
    public void Load_MonoWav_ReturnsExpectedShapeAndRange()
    {
        const int rate = 8000;
        string path = WriteWav(Sine(rate, rate, 440f), rate, channels: 1);

        AudioClip clip = AudioFiles.Load(path);

        Assert.Equal(rate, clip.SampleRate);
        Assert.Equal(1, clip.Channels);
        Assert.Equal(rate, clip.Samples.Length);
        Assert.Equal(rate, clip.FrameCount);
        Assert.Equal(1.0, clip.DurationSeconds, 3);
        Assert.All(clip.Samples.Buffer, s => Assert.InRange(s, -1.0001f, 1.0001f));
    }

    [Fact]
    public void Load_StereoWav_MixesDownToMonoByDefault()
    {
        const int frames = 100;
        var interleaved = new float[frames * 2];
        for (int f = 0; f < frames; f++)
        {
            interleaved[f * 2] = 0.5f;       // left
            interleaved[f * 2 + 1] = -0.5f;  // right
        }
        string path = WriteWav(interleaved, sampleRate: 16000, channels: 2);

        AudioClip clip = AudioFiles.Load(path); // Mono = true by default

        Assert.Equal(1, clip.Channels);
        Assert.Equal(frames, clip.Samples.Length);
        // (0.5 + -0.5) / 2 == 0 exactly for IEEE-float WAV.
        Assert.All(clip.Samples.Buffer, s => Assert.Equal(0f, s, 1e-6f));
    }

    [Fact]
    public void Load_StereoWav_KeepsChannelsWhenMonoDisabled()
    {
        const int frames = 100;
        var interleaved = new float[frames * 2];
        for (int f = 0; f < frames; f++)
        {
            interleaved[f * 2] = 0.25f;
            interleaved[f * 2 + 1] = -0.75f;
        }
        string path = WriteWav(interleaved, sampleRate: 16000, channels: 2);

        AudioClip clip = AudioFiles.Load(path, new AudioLoadOptions { Mono = false });

        Assert.Equal(2, clip.Channels);
        Assert.Equal(frames * 2, clip.Samples.Length);
        Assert.Equal(frames, clip.FrameCount);
        Assert.Equal(0.25f, clip.Samples[0], 1e-6f);
        Assert.Equal(-0.75f, clip.Samples[1], 1e-6f);
    }

    [Fact]
    public void Load_Resample_ChangesRateAndLength()
    {
        const int srcRate = 16000;
        string path = WriteWav(Sine(srcRate, srcRate, 220f), srcRate, channels: 1);

        AudioClip clip = AudioFiles.Load(path, new AudioLoadOptions { TargetSampleRate = 8000 });

        Assert.Equal(8000, clip.SampleRate);
        // 1 s of audio resampled to 8 kHz ≈ 8000 samples (allow filter slack).
        Assert.InRange(clip.Samples.Length, 8000 - 256, 8000 + 256);
        Assert.Equal(1.0, clip.DurationSeconds, 1);
    }

    [Fact]
    public void Load_Normalize_ScalesPeakToOne()
    {
        const int rate = 8000;
        string path = WriteWav(Sine(rate, rate, 440f, amplitude: 0.2f), rate, channels: 1);

        AudioClip clip = AudioFiles.Load(path, new AudioLoadOptions { Normalize = true });

        float peak = 0f;
        foreach (float s in clip.Samples.Buffer)
            peak = MathF.Max(peak, MathF.Abs(s));
        Assert.Equal(1.0f, peak, 0.02f);
    }

    [Fact]
    public void LoadSamples_ReturnsDenseTensorOfSamples()
    {
        const int rate = 4000;
        string path = WriteWav(Sine(rate, rate, 100f), rate, channels: 1);

        DenseTensor<float> samples = AudioFiles.LoadSamples(path);

        Assert.Equal(1, samples.Rank);
        Assert.Equal(rate, samples.Length);
    }

    [Fact]
    public void Load_MissingFile_Throws()
    {
        string path = Path.Combine(_tmp, "does_not_exist.wav");
        Assert.Throws<FileNotFoundException>(() => AudioFiles.Load(path));
    }

    [Fact]
    public void Load_EmptyOrNullPath_Throws()
    {
        Assert.Throws<ArgumentException>(() => AudioFiles.Load(""));
        Assert.Throws<ArgumentNullException>(() => AudioFiles.Load(null!));
    }
}
