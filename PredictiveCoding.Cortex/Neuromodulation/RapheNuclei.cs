namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// RapheNuclei  —  serotonergic time-horizon / patience controller
//
// Doya (2002) maps serotonin onto the time scale of reward integration: higher
// serotonin lengthens the effective horizon (a larger temporal-discount factor γ),
// supporting patience for delayed reward; low serotonin shortens it (impulsivity).
//
// A minimal controller: it holds a serotonin tone in [0,1] and maps it to an
// effective discount factor for the DopaminergicSystem (or any TD learner).
//
// References: Doya (2002); Cools, Nakamura & Daw (2011).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class RapheNuclei
{
    private float _level;

    /// <summary>Serotonin tone in [0,1]; higher = more patience / longer horizon. Clamped on set.</summary>
    public float Level
    {
        get => _level;
        set => _level = Math.Clamp(value, 0f, 1f);
    }

    public RapheNuclei(float level = 0.5f) => Level = level;

    /// <summary>
    /// Effective temporal-discount factor γ ∈ [minGamma, maxGamma], increasing with
    /// serotonin tone: γ = minGamma + Level·(maxGamma − minGamma).
    /// </summary>
    public float DiscountFactor(float minGamma = 0.5f, float maxGamma = 0.99f)
    {
        if (minGamma is < 0f or > 1f) throw new ArgumentOutOfRangeException(nameof(minGamma), "minGamma must be in [0,1].");
        if (maxGamma is < 0f or > 1f) throw new ArgumentOutOfRangeException(nameof(maxGamma), "maxGamma must be in [0,1].");
        if (maxGamma < minGamma)      throw new ArgumentException("maxGamma must be ≥ minGamma.", nameof(maxGamma));
        return minGamma + _level * (maxGamma - minGamma);
    }

    /// <summary>Approximate reward time-horizon implied by the current tone: 1 / (1 − γ).</summary>
    public float TimeHorizon(float minGamma = 0.5f, float maxGamma = 0.99f)
        => 1f / (1f - DiscountFactor(minGamma, maxGamma));
}
