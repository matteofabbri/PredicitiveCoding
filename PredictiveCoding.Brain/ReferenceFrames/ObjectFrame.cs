namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// ObjectFrame  —  object-centric reference frame.
//
// Stores a set of (location, feature) observation pairs for a single object,
// all expressed in the *object's own* reference frame (not the room or world
// frame).  During inference, VotesFor counts how many stored points are
// consistent with a new observation via SDR overlap, giving the CorticalColumn
// evidence to narrow its hypothesis set.
//
// Reference: Hawkins et al. (2019) "A Framework for Intelligence and Cortical
// Function Based on Grid Cells in the Neocortex." Frontiers in Neural Circuits.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class ObjectFrame
{
    private readonly List<(Sdr Location, Sdr Feature)> _points = [];
    private readonly int _minOverlap;

    /// <summary>Caller-assigned identifier for this object.</summary>
    public int ObjectId { get; }

    /// <summary>Number of stored (location, feature) observation pairs.</summary>
    public int PointCount => _points.Count;

    /// <summary>All stored (location, feature) pairs (read-only view).</summary>
    public IReadOnlyList<(Sdr Location, Sdr Feature)> AllPoints => _points;

    /// <param name="objectId">Identifier assigned by the caller (e.g. ObjectMemory).</param>
    /// <param name="minOverlap">Minimum SDR overlap for a location or feature to count as matching.</param>
    public ObjectFrame(int objectId, int minOverlap = 3)
    {
        if (minOverlap < 1) throw new ArgumentOutOfRangeException(nameof(minOverlap));
        ObjectId = objectId;
        _minOverlap = minOverlap;
    }

    /// <summary>Records that <paramref name="feature"/> was sensed at <paramref name="location"/>
    /// on this object.</summary>
    public void AddPoint(Sdr location, Sdr feature)
    {
        ArgumentNullException.ThrowIfNull(location);
        ArgumentNullException.ThrowIfNull(feature);
        _points.Add((location, feature));
    }

    /// <summary>Number of stored points that match both <paramref name="location"/> and
    /// <paramref name="feature"/> within the configured minimum overlap.
    /// Higher → stronger evidence this is the correct object.</summary>
    public int VotesFor(Sdr location, Sdr feature)
    {
        ArgumentNullException.ThrowIfNull(location);
        ArgumentNullException.ThrowIfNull(feature);
        int count = 0;
        foreach (var (loc, feat) in _points)
        {
            if (loc.Size == location.Size && feat.Size == feature.Size
                && loc.Matches(location, _minOverlap)
                && feat.Matches(feature, _minOverlap))
                count++;
        }
        return count;
    }

    /// <summary>Features stored at locations matching <paramref name="queryLocation"/>
    /// within the configured minimum overlap.</summary>
    public IReadOnlyList<Sdr> MatchingFeatures(Sdr queryLocation)
    {
        ArgumentNullException.ThrowIfNull(queryLocation);
        var result = new List<Sdr>();
        foreach (var (loc, feat) in _points)
            if (loc.Size == queryLocation.Size && loc.Matches(queryLocation, _minOverlap))
                result.Add(feat);
        return result;
    }

    /// <summary>Locations at which features matching <paramref name="queryFeature"/>
    /// were observed.</summary>
    public IReadOnlyList<Sdr> MatchingLocations(Sdr queryFeature)
    {
        ArgumentNullException.ThrowIfNull(queryFeature);
        var result = new List<Sdr>();
        foreach (var (loc, feat) in _points)
            if (feat.Size == queryFeature.Size && feat.Matches(queryFeature, _minOverlap))
                result.Add(loc);
        return result;
    }
}
