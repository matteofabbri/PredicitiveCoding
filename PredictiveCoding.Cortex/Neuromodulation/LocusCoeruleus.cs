using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// LocusCoeruleus  —  noradrenergic global-gain / network-reset controller
//
// The locus coeruleus (LC) broadcasts noradrenaline across the cortex.  In the
// adaptive-gain theory (Aston-Jones & Cohen 2005) a *phasic* LC burst transiently
// boosts gain everywhere; Yu & Dayan (2005) cast noradrenaline as the report of
// *unexpected* uncertainty, and Bouret & Sara (2005) as a network *reset* that
// interrupts the current model when the world changes abruptly.
//
// Here the LC watches a single global surprise scalar (e.g. the summed free energy
// of all areas).  It keeps an EMA of the expected surprise; when the current
// surprise exceeds it by PhasicThreshold× it fires a phasic burst and applies a high
// (PhasicGain) precision broadly — a reset — otherwise the resting TonicGain.
//
// Contrast: Thalamus allocates precision *differentially* by reliability;
// LocusCoeruleus applies one gain *uniformly* and only on a surprise spike.
//
// References: Aston-Jones & Cohen (2005); Yu & Dayan (2005); Bouret & Sara (2005).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class LocusCoeruleus
{
    /// <summary>EMA rate of the expected-surprise baseline ∈ (0,1].</summary>
    public float Adaptation { get; }

    /// <summary>Surprise / expected-surprise ratio above which a phasic burst fires (> 1).</summary>
    public float PhasicThreshold { get; }

    /// <summary>Precision (gain) applied broadly during a phasic burst.</summary>
    public float PhasicGain { get; }

    /// <summary>Resting precision (gain) applied when no burst fires.</summary>
    public float TonicGain { get; }

    private float _expectedSurprise;
    private bool  _initialised;

    /// <summary>Current EMA of the expected global surprise.</summary>
    public float ExpectedSurprise => _expectedSurprise;

    public LocusCoeruleus(
        float adaptation = 0.1f, float phasicThreshold = 1.5f,
        float phasicGain = 4f, float tonicGain = 1f)
    {
        if (adaptation is <= 0f or > 1f)
            throw new ArgumentOutOfRangeException(nameof(adaptation), "Adaptation must be in (0,1].");
        if (phasicThreshold <= 1f)
            throw new ArgumentOutOfRangeException(nameof(phasicThreshold), "PhasicThreshold must be > 1.");
        if (phasicGain <= 0f)
            throw new ArgumentOutOfRangeException(nameof(phasicGain), "PhasicGain must be > 0.");
        if (tonicGain <= 0f)
            throw new ArgumentOutOfRangeException(nameof(tonicGain), "TonicGain must be > 0.");

        Adaptation      = adaptation;
        PhasicThreshold = phasicThreshold;
        PhasicGain      = phasicGain;
        TonicGain       = tonicGain;
    }

    /// <summary>
    /// Feeds the current global surprise: returns whether a phasic burst fires and the
    /// noradrenergic gain to apply (PhasicGain on a burst, else TonicGain), then folds
    /// the surprise into the expected-surprise EMA.  The first call only seeds the
    /// baseline (never bursts).
    /// </summary>
    public (bool burst, float gain) Process(float surprise)
    {
        if (surprise < 0f)
            throw new ArgumentOutOfRangeException(nameof(surprise), "Surprise must be ≥ 0.");

        bool burst;
        if (!_initialised)
        {
            _expectedSurprise = surprise;
            _initialised      = true;
            burst             = false;
        }
        else
        {
            burst = surprise > PhasicThreshold * _expectedSurprise;
            _expectedSurprise += Adaptation * (surprise - _expectedSurprise);
        }
        return (burst, burst ? PhasicGain : TonicGain);
    }

    /// <summary>
    /// Reads the global surprise (summed free energy) of the areas, runs <see cref="Process"/>,
    /// and writes the resulting gain as a flat top-layer precision onto every area — a broad
    /// noradrenergic reset on a burst.  Returns the applied gain.
    /// </summary>
    public float Modulate(IReadOnlyList<Network> areas)
    {
        float surprise = GlobalSurprise(areas);
        var (_, gain)  = Process(surprise);
        for (int i = 0; i < areas.Count; i++)
            areas[i].SetTopLayerPrior(null, gain);
        return gain;
    }

    /// <summary>Sum of the current free energies of all areas — the global surprise scalar.</summary>
    public static float GlobalSurprise(IReadOnlyList<Network> areas)
    {
        float s = 0f;
        for (int i = 0; i < areas.Count; i++) s += areas[i].ComputeEnergy();
        return s;
    }

    /// <summary>Clears the baseline; the next <see cref="Process"/> re-seeds it.</summary>
    public void Reset() { _expectedSurprise = 0f; _initialised = false; }
}
