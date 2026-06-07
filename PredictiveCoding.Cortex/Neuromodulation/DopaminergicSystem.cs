namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// DopaminergicSystem  —  midbrain dopamine (VTA / substantia nigra pars compacta)
//
// Phasic dopamine encodes a reward-prediction-error (RPE): the temporal-difference
// signal δ = r + γ·V(s′) − V(s) (Schultz, Dayan & Montague 1997).  Tonic dopamine
// tracks the average reward rate and is the baseline against which surprises are
// measured (Niv et al. 2007).  In a predictive-coding system the RPE gates
// plasticity: a surprising outcome (large |δ|) warrants faster learning.
//
// Pure controller (no persistence): it computes the RPE, maintains the running
// average-reward baseline, and turns an RPE into a modulated learning rate.  Wiring
// the modulated rate into a learner is the caller's responsibility (Network exposes
// no public learning-rate setter — by design this layer only emits the signal).
//
// References: Schultz, Dayan & Montague (1997); Niv et al. (2007); Friston (2009).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class DopaminergicSystem
{
    /// <summary>Temporal discount factor γ ∈ [0,1] applied to the successor value.</summary>
    public float Discount { get; }

    /// <summary>Tonic dopamine floor added to |RPE| when modulating plasticity (≥ 0).</summary>
    public float TonicLevel { get; }

    /// <summary>Learning rate α ∈ (0,1] of the running average-reward baseline.</summary>
    public float BaselineLearningRate { get; }

    private float _baseline;   // running average reward (tonic-DA target, Niv 2007)

    /// <summary>Current average-reward baseline (the tonic dopamine level).</summary>
    public float Baseline => _baseline;

    public DopaminergicSystem(
        float discount = 0.95f, float tonicLevel = 0.1f, float baselineLearningRate = 0.05f)
    {
        if (discount is < 0f or > 1f)
            throw new ArgumentOutOfRangeException(nameof(discount), "Discount must be in [0,1].");
        if (tonicLevel < 0f)
            throw new ArgumentOutOfRangeException(nameof(tonicLevel), "TonicLevel must be ≥ 0.");
        if (baselineLearningRate is <= 0f or > 1f)
            throw new ArgumentOutOfRangeException(nameof(baselineLearningRate), "BaselineLearningRate must be in (0,1].");

        Discount             = discount;
        TonicLevel           = tonicLevel;
        BaselineLearningRate = baselineLearningRate;
    }

    /// <summary>
    /// Phasic reward-prediction-error with an explicit successor value:
    /// δ = reward + γ·valueNext − valueCurrent (Schultz 1997).
    /// </summary>
    public float RewardPredictionError(float reward, float valueCurrent, float valueNext)
        => reward + Discount * valueNext - valueCurrent;

    /// <summary>Terminal reward-prediction-error (no successor): δ = reward − valueCurrent.</summary>
    public float RewardPredictionError(float reward, float valueCurrent)
        => reward - valueCurrent;

    /// <summary>
    /// One average-reward step: returns δ = reward − baseline and moves the baseline
    /// toward the observed reward (tonic dopamine tracking, Niv 2007).
    /// </summary>
    public float Step(float reward)
    {
        float delta = reward - _baseline;
        _baseline += BaselineLearningRate * delta;
        return delta;
    }

    /// <summary>
    /// Modulates a base learning rate by dopamine: η_eff = baseLearningRate·(TonicLevel + |rpe|).
    /// A larger surprise ⇒ more plasticity.
    /// </summary>
    public float ModulateLearningRate(float baseLearningRate, float rpe)
    {
        if (baseLearningRate < 0f)
            throw new ArgumentOutOfRangeException(nameof(baseLearningRate), "Base learning rate must be ≥ 0.");
        return baseLearningRate * (TonicLevel + MathF.Abs(rpe));
    }

    /// <summary>Resets the average-reward baseline to zero.</summary>
    public void Reset() => _baseline = 0f;
}
