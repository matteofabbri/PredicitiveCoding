namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// Habenula  —  the "anti-reward" hub (negative prediction error)
//
// The lateral habenula is the mirror image of the dopaminergic reward signal: its
// neurons fire when an outcome is *worse* than expected and, through the RMTg,
// inhibit midbrain dopamine. It carries the negative reward-prediction-error that
// `DopaminergicSystem` does not, and suppresses dopamine-driven plasticity after
// disappointment. A pure controller paired with `DopaminergicSystem`.
//
// References: Matsumoto & Hikosaka (2007) lateral habenula & negative reward;
//             Hikosaka (2010) habenula in the reward circuit.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class Habenula
{
    /// <summary>How strongly the negative prediction error suppresses dopamine (≥ 0).</summary>
    public float SuppressionGain { get; }

    public Habenula(float suppressionGain = 1f)
    {
        if (suppressionGain < 0f) throw new ArgumentOutOfRangeException(nameof(suppressionGain));
        SuppressionGain = suppressionGain;
    }

    /// <summary>Signed disappointment: expected − reward (positive when worse than expected).</summary>
    public float Disappointment(float reward, float expected) => expected - reward;

    /// <summary>The (non-negative) negative prediction error — how much worse than expected.</summary>
    public float NegativePredictionError(float reward, float expected)
        => MathF.Max(0f, expected - reward);

    /// <summary>Suppresses a dopamine signal by the negative prediction error:
    /// dopamine − gain·max(0, expected − reward). A worse-than-expected outcome lowers
    /// the effective dopamine; a met or exceeded expectation leaves it unchanged.</summary>
    public float SuppressDopamine(float dopamine, float reward, float expected)
        => dopamine - SuppressionGain * NegativePredictionError(reward, expected);
}
