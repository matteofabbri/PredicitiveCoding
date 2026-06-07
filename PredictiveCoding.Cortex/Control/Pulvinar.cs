namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// Pulvinar  —  higher-order thalamic attention hub (top-down precision routing)
//
// The pulvinar is the thalamus's attention nucleus: rather than gating a single
// sensory relay it coordinates *cortico-cortical* communication, boosting the gain
// of the currently attended areas. Where `Thalamus` allocates precision by
// reliability (low free energy), the pulvinar allocates it by *attentional priority*
// (top-down salience): a soft-max over salience distributes a fixed precision budget,
// so the attended area gets the most gain while the mean stays at BasePrecision.
//
// References: Saalmann & Kastner (2011) pulvinar & attention; Shipp (2003) pulvinar
//             as cortical coordinator; Friston (2005) precision = gain.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class Pulvinar
{
    /// <summary>Soft-max temperature over salience (smaller ⇒ winner-take-all).</summary>
    public float Temperature { get; }

    /// <summary>Average precision; the per-area gains sum to BasePrecision·N.</summary>
    public float BasePrecision { get; }

    public Pulvinar(float temperature = 1f, float basePrecision = 1f)
    {
        if (temperature <= 0f) throw new ArgumentOutOfRangeException(nameof(temperature));
        if (basePrecision <= 0f) throw new ArgumentOutOfRangeException(nameof(basePrecision));
        Temperature = temperature;
        BasePrecision = basePrecision;
    }

    /// <summary>Allocates a precision (gain) to each area from its attentional salience:
    /// more salient areas receive more gain. The returned precisions average to
    /// <see cref="BasePrecision"/>.</summary>
    public float[] Allocate(IReadOnlyList<float> saliencePerArea)
    {
        ArgumentNullException.ThrowIfNull(saliencePerArea);
        int n = saliencePerArea.Count;
        if (n == 0) return [];

        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++) max = MathF.Max(max, saliencePerArea[i]);

        var gates = new float[n];
        float sum = 0f;
        for (int i = 0; i < n; i++)
        {
            gates[i] = MathF.Exp((saliencePerArea[i] - max) / Temperature);
            sum += gates[i];
        }

        var precisions = new float[n];
        for (int i = 0; i < n; i++) precisions[i] = BasePrecision * n * (gates[i] / sum);
        return precisions;
    }
}
