namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// NucleusAccumbens  —  ventral striatum: reward → action gating ("wanting")
//
// The nucleus accumbens is the limbic-motor interface: it converts the values of
// available actions into motivational "incentive salience" and gates which one is
// pursued. Phasic dopamine amplifies wanting — the same option is pursued more
// vigorously when dopamine is high — so here incentive = value·(1 + dopamine) and a
// soft-max over incentives gives the action-gating weights. Pairs with
// `DopaminergicSystem` (the dopamine signal) and `BasalGanglia` (the action gate).
//
// References: Berridge & Robinson (1998) incentive salience; Mogenson et al. (1980)
//             limbic-motor interface.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class NucleusAccumbens
{
    /// <summary>Soft-max temperature over incentive salience.</summary>
    public float Temperature { get; }

    public NucleusAccumbens(float temperature = 1f)
    {
        if (temperature <= 0f) throw new ArgumentOutOfRangeException(nameof(temperature));
        Temperature = temperature;
    }

    /// <summary>Incentive salience of each action: value·(1 + dopamine). Dopamine is
    /// clamped at −1 so the amplifier stays non-negative.</summary>
    public float[] IncentiveSalience(IReadOnlyList<float> actionValues, float dopamine)
    {
        ArgumentNullException.ThrowIfNull(actionValues);
        float amp = 1f + MathF.Max(-1f, dopamine);
        var inc = new float[actionValues.Count];
        for (int i = 0; i < inc.Length; i++) inc[i] = actionValues[i] * amp;
        return inc;
    }

    /// <summary>Action-gating weights: soft-max over incentive salience. Higher dopamine
    /// sharpens the distribution toward the best action (more vigorous pursuit).</summary>
    public float[] GateWeights(IReadOnlyList<float> actionValues, float dopamine)
    {
        var inc = IncentiveSalience(actionValues, dopamine);
        int n = inc.Length;
        if (n == 0) return [];

        float max = float.NegativeInfinity;
        for (int i = 0; i < n; i++) max = MathF.Max(max, inc[i]);

        var w = new float[n];
        float sum = 0f;
        for (int i = 0; i < n; i++) { w[i] = MathF.Exp((inc[i] - max) / Temperature); sum += w[i]; }
        for (int i = 0; i < n; i++) w[i] /= sum;
        return w;
    }

    /// <summary>The selected action: the one with the highest incentive salience.</summary>
    public int Select(IReadOnlyList<float> actionValues, float dopamine)
    {
        var inc = IncentiveSalience(actionValues, dopamine);
        int best = -1;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < inc.Length; i++)
            if (inc[i] > bestVal) { bestVal = inc[i]; best = i; }
        return best;
    }
}
