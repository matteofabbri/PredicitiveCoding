using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// Thalamus  —  precision / gain controller across cortical areas
//
// The thalamus (with the reticular nucleus) is widely cast as the brain's
// precision / attention switchboard: it sets the gain on each cortical area's
// prediction errors.  In predictive coding precision IS the gain.  This
// controller reads each area's current free energy and allocates precision
// inversely — areas whose predictions currently fit best (low energy, most
// reliable) get the highest gain — then writes it onto each area's top layer.
//
// References: Friston (2005) precision-weighting; Kanai et al. (2015) thalamic gating.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class Thalamus
{
    /// <summary>Softmax temperature for precision allocation (β over −energy).</summary>
    public float Temperature { get; }

    /// <summary>Average precision level; the per-area gains sum to BasePrecision·N.</summary>
    public float BasePrecision { get; }

    public Thalamus(float temperature = 1f, float basePrecision = 1f)
    {
        if (temperature <= 0f)   throw new ArgumentOutOfRangeException(nameof(temperature));
        if (basePrecision <= 0f) throw new ArgumentOutOfRangeException(nameof(basePrecision));
        Temperature   = temperature;
        BasePrecision = basePrecision;
    }

    /// <summary>
    /// Allocates a precision (gain) to each area from its energy: reliable areas
    /// (low energy) receive more.  Returns the per-area precisions; their mean is
    /// <see cref="BasePrecision"/>.
    /// </summary>
    public float[] Allocate(IReadOnlyList<float> areaEnergies)
    {
        int n = areaEnergies.Count;
        if (n == 0) return [];

        var energies   = areaEnergies as float[] ?? areaEnergies.ToArray();
        var gates      = new float[n];
        MixtureOfExperts.ComputeGates(energies, gates, Temperature, minGate: 0f);

        var precisions = new float[n];
        for (int i = 0; i < n; i++) precisions[i] = BasePrecision * n * gates[i];
        return precisions;
    }

    /// <summary>
    /// Reads the current free energy of each area, allocates precision, and writes
    /// it onto each area's top layer (flat prior at the allocated gain).  Returns
    /// the applied precisions.  Call between inference passes to steer attention.
    /// </summary>
    public float[] Modulate(IReadOnlyList<Network> areas)
    {
        var energies   = new float[areas.Count];
        for (int i = 0; i < areas.Count; i++) energies[i] = areas[i].ComputeEnergy();

        var precisions = Allocate(energies);
        for (int i = 0; i < areas.Count; i++)
            areas[i].SetTopLayerPrior(null, precisions[i]);
        return precisions;
    }
}
