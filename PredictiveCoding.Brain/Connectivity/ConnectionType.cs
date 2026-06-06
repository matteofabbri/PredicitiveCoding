namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// ConnectionType  —  functional role of a long-range cortico-cortical edge.
//
// The Thousand Brains heterarchy uses connections in all four directions,
// not just the classical feedforward hierarchy (V1→V2→…→IT).
//
// Reference: Hawkins, Leadholm & Clay (2025) "Hierarchy or Heterarchy?
// A Theory of Long-Range Connections for the Sensorimotor Brain." (P8)
// ─────────────────────────────────────────────────────────────────────────────

public enum ConnectionType
{
    /// <summary>Classic bottom-up signal from a lower to a higher cortical area
    /// (e.g. V1 → V2 → IT). Carries feature representations upward.</summary>
    FeedForward,

    /// <summary>Top-down prediction from a higher to a lower area.
    /// Carries object / context hypotheses downward to bias feature processing.</summary>
    FeedBack,

    /// <summary>Horizontal connection between areas at the same level of a
    /// functional hierarchy (equivalent to the Phase 5 lateral vote, but
    /// spanning distinct cortical areas rather than columns within one area).</summary>
    Lateral,

    /// <summary>Unconstrained long-range connection whose direction does not fit
    /// the strict feedforward / feedback / lateral categories — the defining
    /// feature of a heterarchy. Carries hypothesis constraints in any direction.</summary>
    Heterarchical,
}
