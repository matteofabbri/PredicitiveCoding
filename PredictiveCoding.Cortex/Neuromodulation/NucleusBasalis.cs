using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// NucleusBasalis  —  cholinergic expected-uncertainty controller
//
// The nucleus basalis of Meynert supplies acetylcholine to the cortex.  Yu & Dayan
// (2005) identify acetylcholine with *expected* uncertainty — the known, slowly-
// varying unreliability of a context.  High acetylcholine biases processing toward
// bottom-up sensory evidence and raises plasticity, so the system relearns the parts
// of its model it already knows to be unreliable (Hasselmo 2006).
//
// This pure controller keeps a per-area EMA of expected uncertainty (each area's own
// free energy) and turns it into a per-area acetylcholine level and a per-area
// modulated learning rate.  Contrast: Thalamus sets precision/gain, LocusCoeruleus
// is global·phasic·unexpected; NucleusBasalis is per-area·tonic·expected.
//
// References: Yu & Dayan (2005); Hasselmo (2006).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class NucleusBasalis
{
    /// <summary>Number of cortical areas tracked.</summary>
    public int AreaCount { get; }

    /// <summary>EMA rate of each area's expected-uncertainty estimate ∈ (0,1].</summary>
    public float Adaptation { get; }

    /// <summary>Scales expected uncertainty into an acetylcholine level (≥ 0).</summary>
    public float Gain { get; }

    private readonly float[] _expected;
    private bool _initialised;

    /// <summary>Current per-area expected-uncertainty EMA.</summary>
    public IReadOnlyList<float> ExpectedUncertainty => _expected;

    public NucleusBasalis(int areaCount, float adaptation = 0.1f, float gain = 1f)
    {
        if (areaCount <= 0)
            throw new ArgumentOutOfRangeException(nameof(areaCount), "AreaCount must be ≥ 1.");
        if (adaptation is <= 0f or > 1f)
            throw new ArgumentOutOfRangeException(nameof(adaptation), "Adaptation must be in (0,1].");
        if (gain < 0f)
            throw new ArgumentOutOfRangeException(nameof(gain), "Gain must be ≥ 0.");

        AreaCount  = areaCount;
        Adaptation = adaptation;
        Gain       = gain;
        _expected  = new float[areaCount];
    }

    /// <summary>
    /// Folds the current per-area free energies into the expected-uncertainty EMA and
    /// returns the per-area acetylcholine level ACh[i] = Gain · expectedUncertainty[i].
    /// The first call seeds the EMA directly from the supplied energies.
    /// </summary>
    public float[] Update(IReadOnlyList<float> areaEnergies)
    {
        if (areaEnergies.Count != AreaCount)
            throw new ArgumentException($"Expected {AreaCount} energies, got {areaEnergies.Count}.", nameof(areaEnergies));

        var ach = new float[AreaCount];
        for (int i = 0; i < AreaCount; i++)
        {
            float e = areaEnergies[i];
            _expected[i] = _initialised ? _expected[i] + Adaptation * (e - _expected[i]) : e;
            ach[i] = Gain * _expected[i];
        }
        _initialised = true;
        return ach;
    }

    /// <summary>Reads the areas' free energies and runs <see cref="Update(IReadOnlyList{float})"/>.</summary>
    public float[] Update(IReadOnlyList<Network> areas)
    {
        var energies = new float[areas.Count];
        for (int i = 0; i < areas.Count; i++) energies[i] = areas[i].ComputeEnergy();
        return Update(energies);
    }

    /// <summary>
    /// Per-area plasticity modulation: η_eff[i] = baseLearningRate·(1 + ACh[i]).
    /// Higher expected uncertainty ⇒ faster relearning of that area.
    /// </summary>
    public float[] ModulateLearningRates(float baseLearningRate, IReadOnlyList<float> areaEnergies)
    {
        if (baseLearningRate < 0f)
            throw new ArgumentOutOfRangeException(nameof(baseLearningRate), "Base learning rate must be ≥ 0.");

        var ach   = Update(areaEnergies);
        var rates = new float[AreaCount];
        for (int i = 0; i < AreaCount; i++) rates[i] = baseLearningRate * (1f + ach[i]);
        return rates;
    }

    /// <summary>Clears the EMA; the next <see cref="Update(IReadOnlyList{float})"/> re-seeds it.</summary>
    public void Reset() { Array.Clear(_expected); _initialised = false; }
}
