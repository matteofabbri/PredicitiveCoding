namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// ThousandBrainsConfig  —  configuration record for ThousandBrainsSystem.
//
// Specifies the cortical topology, column parameters, and algorithm settings
// used when constructing a ThousandBrainsSystem via Create().
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Immutable configuration for a <see cref="ThousandBrainsSystem"/>.</summary>
public sealed record ThousandBrainsConfig
{
    /// <summary>Number of independent sensory cortical areas.  Each area hosts
    /// <see cref="ColumnsPerArea"/> cortical columns. Minimum 1.</summary>
    public required int SensoryAreas { get; init; }

    /// <summary>Column-level configuration shared (with per-area seed offsets) by
    /// all cortical columns across all areas.</summary>
    public required CorticalColumnConfig ColumnConfig { get; init; }

    /// <summary>Columns per cortical area.  Must be ≥ 2 (MultiColumnSystem constraint).
    /// Default 2.</summary>
    public int ColumnsPerArea { get; init; } = 2;

    /// <summary>Lateral voting topology used within each cortical area.
    /// Default <see cref="VotingTopology.AllToAll"/>.</summary>
    public VotingTopology IntraAreaTopology { get; init; } = VotingTopology.AllToAll;

    /// <summary>Long-range connection topology between cortical areas.
    /// Default <see cref="VotingTopology.AllToAll"/>.</summary>
    public VotingTopology InterAreaTopology { get; init; } = VotingTopology.AllToAll;

    /// <summary>Maximum inter-area message-passing rounds per observation.
    /// Default 6.</summary>
    public int MaxInferenceRounds { get; init; } = 6;

    /// <summary>Minimum consensus agreement fraction within each area.
    /// Passed to each <see cref="MultiColumnSystem"/>. Default 0.5.</summary>
    public float MinAgreementFraction { get; init; } = 0.5f;

    /// <summary>Minimum source agreement score for a hypothesis to propagate
    /// along a long-range connection. Default 0.0 (propagate all).</summary>
    public float ConvergenceThreshold { get; init; } = 0.0f;

    /// <summary>Master RNG seed; per-area and per-column seeds are derived from
    /// this value. Default 42.</summary>
    public int Seed { get; init; } = 42;
}
