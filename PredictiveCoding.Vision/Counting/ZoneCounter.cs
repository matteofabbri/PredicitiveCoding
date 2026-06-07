using PredictiveCoding.Vision.Tracking;

namespace PredictiveCoding.Vision.Counting;

/// <summary>
/// Per-region occupancy and dwell. Given a polygonal zone, it reports how many
/// tracks are inside <b>right now</b> (<see cref="Occupancy"/>), how many distinct
/// tracks have <b>ever entered</b> (<see cref="Entered"/>), and per-track dwell time
/// in frames. A track's ground position (the box's bottom-centre = its feet) is used
/// for the inside test, which is the right anchor for a floor zone seen from above
/// or at an angle.
/// </summary>
public sealed class ZoneCounter
{
    private readonly (float X, float Y)[] _poly;
    private readonly HashSet<int> _everEntered = [];
    private readonly Dictionary<int, int> _dwell = [];

    /// <summary>Tracks inside the zone in the most recent update.</summary>
    public int Occupancy { get; private set; }

    /// <summary>Distinct tracks that have entered the zone at least once.</summary>
    public int Entered => _everEntered.Count;

    /// <param name="polygon">Zone vertices in order (≥ 3); the polygon auto-closes.</param>
    public ZoneCounter(IReadOnlyList<(float X, float Y)> polygon)
    {
        ArgumentNullException.ThrowIfNull(polygon);
        if (polygon.Count < 3)
            throw new ArgumentException("A zone needs at least 3 vertices.", nameof(polygon));
        _poly = [.. polygon];
    }

    /// <summary>Frames <paramref name="id"/> has spent inside the zone (0 if never).</summary>
    public int DwellFrames(int id) => _dwell.GetValueOrDefault(id, 0);

    /// <summary>True if the point lies inside the zone polygon.</summary>
    public bool IsInside(float x, float y)
    {
        bool inside = false;
        for (int i = 0, j = _poly.Length - 1; i < _poly.Length; j = i++)
        {
            var (xi, yi) = _poly[i];
            var (xj, yj) = _poly[j];
            if (yi > y != yj > y &&
                x < (xj - xi) * (y - yi) / (yj - yi) + xi)
                inside = !inside;
        }
        return inside;
    }

    /// <summary>Updates occupancy/dwell from the current confirmed tracks
    /// (uses each track's feet = box bottom-centre).</summary>
    public void Update(IReadOnlyList<Track> tracks)
    {
        ArgumentNullException.ThrowIfNull(tracks);
        var points = new List<(int, float, float)>(tracks.Count);
        foreach (var t in tracks) points.Add((t.Id, t.CenterX, t.Y2));
        Update(points);
    }

    /// <summary>Updates occupancy/dwell from raw (id, x, y) anchor points.</summary>
    public void Update(IReadOnlyList<(int Id, float X, float Y)> points)
    {
        ArgumentNullException.ThrowIfNull(points);
        int occupancy = 0;
        foreach (var (id, x, y) in points)
        {
            if (!IsInside(x, y)) continue;
            occupancy++;
            _everEntered.Add(id);
            _dwell[id] = _dwell.GetValueOrDefault(id, 0) + 1;
        }
        Occupancy = occupancy;
    }

    /// <summary>Clears occupancy, dwell and entry history.</summary>
    public void Reset()
    {
        _everEntered.Clear();
        _dwell.Clear();
        Occupancy = 0;
    }
}
