using PredictiveCoding.Core;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// InferiorColliculus  —  auditory midbrain front-end (spectral lateral inhibition)
//
// The inferior colliculus is the auditory analogue of the retina/LGN: it sits below
// AuditoryCortex and sharpens the cochleotopic (frequency) representation by lateral
// inhibition across neighbouring frequency channels — subtracting a local prediction
// (the surround on the tonotopic axis) so only the unpredicted spectral contrast is
// passed on. A 1-D center-surround that whitens the spectrum, mirroring `RetinaLgn`
// and `OlfactoryBulb` for the auditory pathway.
//
// References: Suga (1995) tonotopic lateral inhibition; Rao & Ballard (1999)
//             predictive coding front-ends.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class InferiorColliculus
{
    /// <summary>Number of tonotopic (frequency) channels.</summary>
    public int Channels { get; }

    /// <summary>Half-width of the inhibitory surround on the frequency axis.</summary>
    public int SurroundRadius { get; }

    public InferiorColliculus(int channels, int surroundRadius = 2)
    {
        if (channels < 2) throw new ArgumentOutOfRangeException(nameof(channels), "Need ≥ 2 channels.");
        if (surroundRadius < 1) throw new ArgumentOutOfRangeException(nameof(surroundRadius));
        Channels = channels;
        SurroundRadius = surroundRadius;
    }

    /// <summary>Applies frequency-axis lateral inhibition: each channel minus the mean
    /// of its neighbours (excluding itself), whitening the spectrum. A flat spectrum
    /// cancels to ≈0; spectral peaks are sharpened.</summary>
    public DenseTensor<float> Process(DenseTensor<float> spectrum)
    {
        if (spectrum.Length != Channels)
            throw new ArgumentException($"spectrum has {spectrum.Length} channels; expected {Channels}.");

        var buf = new float[Channels];
        for (int i = 0; i < Channels; i++)
        {
            float sum = 0f;
            int n = 0;
            for (int k = -SurroundRadius; k <= SurroundRadius; k++)
            {
                if (k == 0) continue;
                int j = i + k;
                if (j < 0 || j >= Channels) continue;
                sum += spectrum.Buffer[j];
                n++;
            }
            float surround = n > 0 ? sum / n : 0f;
            buf[i] = spectrum.Buffer[i] - surround;
        }
        return new DenseTensor<float>(buf, Channels);
    }
}
