namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// AreaNode  —  graph node representing one cortical area.
//
// Each area wraps a MultiColumnSystem (N columns sharing the same object
// memory and voting protocol) and tracks the consensus the area currently
// holds. The HeterarchicalMessagePassing algorithm updates CurrentOutput
// after each propagation round.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class AreaNode
{
    /// <summary>Unique identifier for this area within the graph.</summary>
    public int AreaId { get; }

    /// <summary>Optional human-readable label (e.g. "V1", "PFC", "S1").</summary>
    public string Name { get; }

    /// <summary>The multi-column sensorimotor system housed in this cortical area.</summary>
    public MultiColumnSystem ColumnSystem { get; }

    /// <summary>Latest consensus computed for this area; null before the first round.</summary>
    public ConsensusResult? CurrentOutput { get; private set; }

    public AreaNode(int areaId, MultiColumnSystem columnSystem, string? name = null)
    {
        ArgumentNullException.ThrowIfNull(columnSystem);
        AreaId = areaId;
        ColumnSystem = columnSystem;
        Name = name ?? $"Area{areaId}";
    }

    /// <summary>Stores a new consensus result as the area's current belief state.
    /// Called by <see cref="HeterarchicalMessagePassing"/> after each round.</summary>
    public void UpdateOutput(ConsensusResult output)
    {
        ArgumentNullException.ThrowIfNull(output);
        CurrentOutput = output;
    }

    public override string ToString() => $"{Name}(id={AreaId})";
}
