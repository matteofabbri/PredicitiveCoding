using PredictiveCoding.Networks.MoE;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// BasalGanglia  —  action selection by energy-based gating (go / no-go)
//
// The basal ganglia arbitrate between candidate actions, releasing the winner
// (Go) and suppressing the rest (No-Go).  Here each candidate action is scored by
// the free energy it would incur (lower = better, supplied by the caller, e.g.
// from a world model or motor cortex).  A softmax over −energy gives the action
// "vigour" distribution; the lowest-energy action is selected, unless even the
// best exceeds a No-Go threshold, in which case action is withheld (−1).
//
// References: Mink (1996) BG action selection; Friston et al. (2017) active inference.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class BasalGanglia
{
    /// <summary>Inverse temperature β of the selection softmax over −energy.</summary>
    public float GatingTemperature { get; }

    /// <summary>If set, actions are withheld when the best energy exceeds this value.</summary>
    public float? NoGoThreshold { get; }

    public BasalGanglia(float gatingTemperature = 1f, float? noGoThreshold = null)
    {
        if (gatingTemperature <= 0f) throw new ArgumentOutOfRangeException(nameof(gatingTemperature));
        GatingTemperature = gatingTemperature;
        NoGoThreshold     = noGoThreshold;
    }

    /// <summary>
    /// Selects an action from per-action free energies.  Returns the chosen action
    /// index (−1 = No-Go) and the gating distribution (action vigour) over actions.
    /// </summary>
    public (int action, float[] gates) Select(IReadOnlyList<float> actionEnergies)
    {
        int n = actionEnergies.Count;
        if (n == 0) throw new ArgumentException("At least one candidate action is required.", nameof(actionEnergies));

        var energies = actionEnergies as float[] ?? actionEnergies.ToArray();
        var gates    = new float[n];
        MixtureOfExperts.ComputeGates(energies, gates, GatingTemperature, minGate: 0f);

        int   best       = 0;
        float bestEnergy = energies[0];
        for (int i = 1; i < n; i++)
            if (energies[i] < bestEnergy) { bestEnergy = energies[i]; best = i; }

        if (NoGoThreshold is { } thr && bestEnergy > thr)
            return (-1, gates);   // No-Go: withhold action
        return (best, gates);
    }
}
