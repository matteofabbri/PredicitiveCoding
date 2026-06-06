namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// ObjectMemory  —  storage for learned sensorimotor object models.
//
// Maps integer object ids to their ObjectFrames — one per learned object — and
// exposes IntersectingObjectIds which returns all ids whose frame has at least
// one (location, feature) point consistent with a query pair.  The CorticalColumn
// calls this after every sensation to intersect the result with the current
// hypothesis set, progressively narrowing to the correct object.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class ObjectMemory
{
    private readonly Dictionary<int, ObjectFrame> _frames = [];
    private readonly int _minOverlap;

    /// <summary>Number of distinct objects stored.</summary>
    public int ObjectCount => _frames.Count;

    /// <summary>Total (location, feature) points summed across all objects.</summary>
    public int PointCount => _frames.Values.Sum(f => f.PointCount);

    /// <param name="minOverlap">Minimum SDR overlap for a point to count as a match.
    /// Forwarded to each <see cref="ObjectFrame"/>.</param>
    public ObjectMemory(int minOverlap = 3)
    {
        if (minOverlap < 1) throw new ArgumentOutOfRangeException(nameof(minOverlap));
        _minOverlap = minOverlap;
    }

    /// <summary>Records that <paramref name="feature"/> was sensed at <paramref name="location"/>
    /// on <paramref name="objectId"/>. Creates the ObjectFrame on first use for that id.</summary>
    public void AddPoint(int objectId, Sdr location, Sdr feature)
    {
        ArgumentNullException.ThrowIfNull(location);
        ArgumentNullException.ThrowIfNull(feature);

        if (!_frames.TryGetValue(objectId, out var frame))
        {
            frame = new ObjectFrame(objectId, _minOverlap);
            _frames[objectId] = frame;
        }
        frame.AddPoint(location, feature);
    }

    /// <summary>Object ids that have at least one stored point matching both
    /// <paramref name="location"/> and <paramref name="feature"/> within the configured overlap.
    /// Used by the CorticalColumn to narrow the active hypothesis set.</summary>
    public IReadOnlyList<int> IntersectingObjectIds(Sdr location, Sdr feature)
    {
        ArgumentNullException.ThrowIfNull(location);
        ArgumentNullException.ThrowIfNull(feature);

        var result = new List<int>();
        foreach (var (id, frame) in _frames)
            if (frame.VotesFor(location, feature) > 0)
                result.Add(id);
        return result;
    }

    /// <summary>All (location, feature) points stored for <paramref name="objectId"/>.
    /// Returns an empty list when the id is unknown.</summary>
    public IReadOnlyList<(Sdr Location, Sdr Feature)> GetPoints(int objectId)
        => _frames.TryGetValue(objectId, out var frame) ? frame.AllPoints : [];

    /// <summary>Removes all objects and their observations.</summary>
    public void Clear() => _frames.Clear();
}
