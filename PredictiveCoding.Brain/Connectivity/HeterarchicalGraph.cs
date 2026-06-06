namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// HeterarchicalGraph  —  directed graph of cortical areas + long-range edges.
//
// Provides a substrate for multi-area message passing that is not constrained
// to any strict hierarchy. Areas can be connected in any combination of
// feedforward, feedback, lateral and heterarchical edges, enabling the
// non-hierarchical loops described in P8.
//
// Reference: Hawkins, Leadholm & Clay (2025) "Hierarchy or Heterarchy?" (P8).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class HeterarchicalGraph
{
    private readonly Dictionary<int, AreaNode> _areas = [];
    private readonly List<LongRangeConnection> _connections = [];
    private int _nextConnectionId;

    /// <summary>Number of cortical areas (nodes) in the graph.</summary>
    public int AreaCount => _areas.Count;

    /// <summary>Number of directed connections (edges) in the graph.</summary>
    public int ConnectionCount => _connections.Count;

    /// <summary>All registered cortical areas.</summary>
    public IReadOnlyCollection<AreaNode> Areas => _areas.Values;

    /// <summary>Adds a cortical area backed by <paramref name="columnSystem"/> to the graph.
    /// Throws if an area with the same id already exists.</summary>
    public AreaNode AddArea(int areaId, MultiColumnSystem columnSystem, string? name = null)
    {
        ArgumentNullException.ThrowIfNull(columnSystem);
        if (_areas.ContainsKey(areaId))
            throw new ArgumentException($"Area {areaId} is already registered.", nameof(areaId));
        var node = new AreaNode(areaId, columnSystem, name);
        _areas[areaId] = node;
        return node;
    }

    /// <summary>Adds a directed long-range connection from <paramref name="sourceId"/> to
    /// <paramref name="targetId"/>. Returns the created edge.</summary>
    public LongRangeConnection Connect(
        int sourceId, int targetId,
        ConnectionType type = ConnectionType.Heterarchical,
        float weight = 1.0f)
    {
        RequireArea(sourceId);
        RequireArea(targetId);
        if (sourceId == targetId)
            throw new ArgumentException("Self-connections are not allowed.");

        var conn = new LongRangeConnection(
            _nextConnectionId++, sourceId, targetId, type, weight);
        _connections.Add(conn);
        return conn;
    }

    /// <summary>Returns the area node for <paramref name="areaId"/>.</summary>
    public AreaNode GetArea(int areaId)
    {
        if (!_areas.TryGetValue(areaId, out var node))
            throw new KeyNotFoundException($"Area {areaId} not found in graph.");
        return node;
    }

    /// <summary>All connections whose source is <paramref name="areaId"/>.</summary>
    public IReadOnlyList<LongRangeConnection> OutgoingConnections(int areaId)
    {
        RequireArea(areaId);
        return _connections.Where(c => c.SourceAreaId == areaId).ToList();
    }

    /// <summary>All connections whose target is <paramref name="areaId"/>.</summary>
    public IReadOnlyList<LongRangeConnection> IncomingConnections(int areaId)
    {
        RequireArea(areaId);
        return _connections.Where(c => c.TargetAreaId == areaId).ToList();
    }

    /// <summary>Removes all areas and connections, resetting the graph to empty.</summary>
    public void Clear()
    {
        _areas.Clear();
        _connections.Clear();
        _nextConnectionId = 0;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private void RequireArea(int areaId)
    {
        if (!_areas.ContainsKey(areaId))
            throw new KeyNotFoundException($"Area {areaId} is not registered in the graph.");
    }
}
