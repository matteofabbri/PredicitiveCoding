using PredictiveCoding.Core;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// OlfactoryBulb  —  decorrelating gain-control front-end to the PiriformCortex
//
// The olfactory bulb sits between the receptors and piriform cortex.  Mitral cells,
// via granule-cell lateral inhibition, perform gain control / decorrelation on the
// raw receptor pattern, normalising contrast before the cortex stores or recognises
// the odour.  This mirrors what RetinaLgn does for vision (a fixed whitening front-
// end), but on a 1-D receptor vector: subtract the global mean and divide by the
// standard deviation.
//
// References: Cleland (2010) early olfactory processing; Atick & Redlich (1992) whitening.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class OlfactoryBulb
{
    public int   Receptors { get; }
    public float Epsilon   { get; }

    public OlfactoryBulb(int receptors, float epsilon = 1e-6f)
    {
        if (receptors < 1) throw new ArgumentOutOfRangeException(nameof(receptors), "receptors must be >= 1.");
        if (epsilon  <= 0f) throw new ArgumentOutOfRangeException(nameof(epsilon), "epsilon must be > 0.");
        Receptors = receptors;
        Epsilon   = epsilon;
    }

    /// <summary>
    /// Decorrelating gain control (lateral inhibition): subtracts the global mean and
    /// divides by the standard deviation, returning a whitened odour vector of the
    /// same length (zero mean, ≈unit variance).  A constant (uninformative) input
    /// maps to ≈0, exactly as center-surround cancels a flat image.
    /// </summary>
    public DenseTensor<float> Process(DenseTensor<float> odor)
    {
        if (odor.Length != Receptors)
            throw new ArgumentException($"odor has {odor.Length} elements; expected {Receptors}.");

        float mean = 0f;
        for (int i = 0; i < Receptors; i++) mean += odor.Buffer[i];
        mean /= Receptors;

        float var = 0f;
        for (int i = 0; i < Receptors; i++) { float d = odor.Buffer[i] - mean; var += d * d; }
        var /= Receptors;
        float inv = 1f / (MathF.Sqrt(var) + Epsilon);

        var outp = new DenseTensor<float>(Receptors);
        for (int i = 0; i < Receptors; i++) outp.Buffer[i] = (odor.Buffer[i] - mean) * inv;
        return outp;
    }
}
