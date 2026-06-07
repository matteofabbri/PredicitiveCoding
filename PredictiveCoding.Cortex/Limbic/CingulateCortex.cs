using PredictiveCoding.Expanded.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// CingulateCortex (ACC)  —  conflict & surprise monitoring
//
// The anterior cingulate monitors for response conflict and errors and signals the
// need for top-down control (engaging the prefrontal cortex).  In predictive-coding
// terms it watches the free energies of competing options/areas: high *surprise*
// (large total energy) and high *conflict* (several options equally good) both call
// for more control.  A pure controller in the style of Thalamus / BasalGanglia.
//
// References: Botvinick et al. (2001) conflict monitoring; Shenhav et al. (2013).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class CingulateCortex
{
    /// <summary>Inverse temperature β of the softmax-over-(−energy) used for conflict.</summary>
    public float Temperature { get; }

    public CingulateCortex(float temperature = 1f)
    {
        if (temperature <= 0f) throw new ArgumentOutOfRangeException(nameof(temperature));
        Temperature = temperature;
    }

    /// <summary>Total free energy across areas/options — the global surprise scalar.</summary>
    public float Surprise(IReadOnlyList<float> energies)
    {
        float s = 0f;
        for (int i = 0; i < energies.Count; i++) s += energies[i];
        return s;
    }

    /// <summary>
    /// Response conflict ∈ [0,1]: the normalised entropy of the softmax-over-(−energy)
    /// distribution over options.  ≈1 when candidates are equally good (high conflict),
    /// ≈0 when one option dominates.
    /// </summary>
    public float Conflict(IReadOnlyList<float> optionEnergies)
    {
        int n = optionEnergies.Count;
        if (n <= 1) return 0f;

        var energies = optionEnergies as float[] ?? optionEnergies.ToArray();
        var gates    = new float[n];
        MixtureOfExperts.ComputeGates(energies, gates, Temperature, minGate: 0f);

        float h = 0f;
        for (int i = 0; i < n; i++)
            if (gates[i] > 0f) h -= gates[i] * MathF.Log(gates[i]);
        return h / MathF.Log(n);   // normalise by the maximum entropy (log n)
    }

    /// <summary>Returns (surprise, conflict) for monitoring.</summary>
    public (float surprise, float conflict) Monitor(IReadOnlyList<float> energies)
        => (Surprise(energies), Conflict(energies));

    /// <summary>True when conflict reaches the threshold ⇒ engage top-down (PFC) control.</summary>
    public bool NeedsControl(IReadOnlyList<float> optionEnergies, float conflictThreshold = 0.8f)
        => Conflict(optionEnergies) >= conflictThreshold;
}
