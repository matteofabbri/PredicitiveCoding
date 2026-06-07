namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// PeriaqueductalGray (PAG)  —  defensive-behaviour selection by threat imminence
//
// The PAG is the brainstem hub that turns the amygdala's threat appraisal into a
// concrete defensive action. Which action depends on *threat imminence* (Fanselow):
// a distant predator triggers freezing (risk assessment), a closer one triggers
// flight, and contact triggers a fight/defensive burst. A pure controller that maps
// (threat magnitude, proximity) → a defensive response; pairs with `Amygdala`.
//
// References: Fanselow (1994) threat-imminence continuum; Bandler & Shipley (1994)
//             PAG columns; Mobbs et al. (2007) human PAG & imminent threat.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>The PAG's selected defensive behaviour.</summary>
public enum DefensiveResponse { None, Freeze, Flight, Fight }

public sealed class PeriaqueductalGray
{
    /// <summary>Threat magnitude below which no defensive response is triggered.</summary>
    public float ThreatThreshold { get; }

    public PeriaqueductalGray(float threatThreshold = 0.1f)
    {
        if (threatThreshold < 0f) throw new ArgumentOutOfRangeException(nameof(threatThreshold));
        ThreatThreshold = threatThreshold;
    }

    /// <summary>Selects a defensive response from threat magnitude and proximity
    /// (0 = distal, 1 = contact). Below threat threshold → None; otherwise distal →
    /// Freeze, intermediate → Flight, near/contact → Fight.</summary>
    public DefensiveResponse Respond(float threat, float proximity)
    {
        if (threat < ThreatThreshold) return DefensiveResponse.None;
        float p = Math.Clamp(proximity, 0f, 1f);
        if (p < 1f / 3f) return DefensiveResponse.Freeze;
        if (p < 2f / 3f) return DefensiveResponse.Flight;
        return DefensiveResponse.Fight;
    }
}
