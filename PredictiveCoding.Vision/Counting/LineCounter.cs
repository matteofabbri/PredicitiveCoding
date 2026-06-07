using PredictiveCoding.Vision.Tracking;

namespace PredictiveCoding.Vision.Counting;

/// <summary>
/// Tripwire / line-crossing counter. Given a virtual line segment A→B, it watches
/// each track's centre between frames and counts a crossing whenever the segment
/// the track travelled actually intersects the line. Direction is split into two
/// tallies: a track that ends on the <b>left</b> of the directed line A→B counts as
/// <see cref="In"/>, on the right as <see cref="Out"/> — so a doorway line gives
/// "entered" vs "left". Off-segment passes (level with, but beyond the ends of, the
/// line) are ignored, which is what makes it a tripwire rather than an infinite line.
/// </summary>
public sealed class LineCounter
{
    private readonly float _ax, _ay, _bx, _by;
    private readonly Dictionary<int, (float X, float Y)> _prev = [];

    /// <summary>Crossings that ended on the left of A→B (e.g. people entering).</summary>
    public int In { get; private set; }

    /// <summary>Crossings that ended on the right of A→B (e.g. people leaving).</summary>
    public int Out { get; private set; }

    /// <summary>Net flow across the line (<see cref="In"/> − <see cref="Out"/>).</summary>
    public int Net => In - Out;

    /// <param name="ax">Line start X.</param><param name="ay">Line start Y.</param>
    /// <param name="bx">Line end X.</param><param name="by">Line end Y.</param>
    public LineCounter(float ax, float ay, float bx, float by)
    {
        if (ax == bx && ay == by)
            throw new ArgumentException("Line endpoints must differ.");
        _ax = ax; _ay = ay; _bx = bx; _by = by;
    }

    /// <summary>Updates with the current confirmed tracks (uses each track's centre).</summary>
    public void Update(IReadOnlyList<Track> tracks)
    {
        ArgumentNullException.ThrowIfNull(tracks);
        var centres = new List<(int, float, float)>(tracks.Count);
        foreach (var t in tracks) centres.Add((t.Id, t.CenterX, t.CenterY));
        Update(centres);
    }

    /// <summary>Updates with raw (id, x, y) track centres for this frame.</summary>
    public void Update(IReadOnlyList<(int Id, float X, float Y)> centres)
    {
        ArgumentNullException.ThrowIfNull(centres);
        foreach (var (id, x, y) in centres)
        {
            if (_prev.TryGetValue(id, out var p)
                && SegmentsIntersect(p.X, p.Y, x, y, _ax, _ay, _bx, _by))
            {
                float side = Cross(_bx - _ax, _by - _ay, x - _ax, y - _ay);
                if (side > 0) In++;
                else if (side < 0) Out++;
            }
            _prev[id] = (x, y);
        }
    }

    /// <summary>Drops the remembered position of a track (e.g. when it retires).</summary>
    public void Forget(int id) => _prev.Remove(id);

    /// <summary>Clears all counts and remembered positions.</summary>
    public void Reset()
    {
        _prev.Clear();
        In = 0;
        Out = 0;
    }

    private static float Cross(float ux, float uy, float vx, float vy) => ux * vy - uy * vx;

    // Proper (non-collinear) segment intersection test: P1P2 vs P3P4.
    private static bool SegmentsIntersect(
        float p1x, float p1y, float p2x, float p2y,
        float p3x, float p3y, float p4x, float p4y)
    {
        float d1 = Cross(p4x - p3x, p4y - p3y, p1x - p3x, p1y - p3y);
        float d2 = Cross(p4x - p3x, p4y - p3y, p2x - p3x, p2y - p3y);
        float d3 = Cross(p2x - p1x, p2y - p1y, p3x - p1x, p3y - p1y);
        float d4 = Cross(p2x - p1x, p2y - p1y, p4x - p1x, p4y - p1y);
        return ((d1 > 0 && d2 < 0) || (d1 < 0 && d2 > 0))
            && ((d3 > 0 && d4 < 0) || (d3 < 0 && d4 > 0));
    }
}
