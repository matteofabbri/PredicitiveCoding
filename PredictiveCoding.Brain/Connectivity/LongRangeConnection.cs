namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// LongRangeConnection  —  directed edge in the heterarchical cortical graph.
//
// Carries hypothesis constraints (and, when areas share the same SDR size,
// active-cell SDRs) from one cortical area to another. The Weight field
// modulates how strongly the source area's current consensus constrains the
// target: a weight of 1.0 applies the source's full candidate set; lower
// weights restrict propagation to candidates the source reports with high
// confidence (AgreementScore ≥ weight).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record LongRangeConnection
{
    /// <summary>Unique identifier within the graph (auto-assigned).</summary>
    public int ConnectionId { get; init; }

    /// <summary>Area that originates the signal.</summary>
    public int SourceAreaId { get; init; }

    /// <summary>Area that receives the signal.</summary>
    public int TargetAreaId { get; init; }

    /// <summary>Functional role of this connection.</summary>
    public ConnectionType Type { get; init; }

    /// <summary>Signal strength in (0, 1]. The target only receives candidates
    /// for which the source's AgreementScore ≥ Weight, so a value of 1.0
    /// propagates only unanimously-agreed candidates.</summary>
    public float Weight { get; init; }

    internal LongRangeConnection(
        int connectionId, int sourceAreaId, int targetAreaId,
        ConnectionType type, float weight)
    {
        if (weight <= 0f || weight > 1f)
            throw new ArgumentOutOfRangeException(nameof(weight), "Weight must be in (0, 1].");
        ConnectionId = connectionId;
        SourceAreaId = sourceAreaId;
        TargetAreaId = targetAreaId;
        Type = type;
        Weight = weight;
    }

    public override string ToString()
        => $"[{ConnectionId}] {SourceAreaId}→{TargetAreaId} ({Type}, w={Weight:F2})";
}
