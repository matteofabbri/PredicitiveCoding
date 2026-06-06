using PredictiveCoding.Vision.Detectors;

namespace PredictiveCoding.Vision.Tracking;

/// <summary>
/// One tracked person followed across frames. The box follows a constant-velocity
/// motion model smoothed by a fixed-gain <b>alpha-beta filter</b> (a lightweight
/// Kalman): each frame the box is advanced by its velocity, then — when a detection
/// is matched — corrected toward the measurement. Lifecycle bookkeeping (hits, age,
/// frames since last match, confirmation) is owned by <see cref="SortTracker"/>.
/// </summary>
public sealed class Track
{
    // Fixed alpha-beta gains: alpha corrects position, beta corrects velocity.
    private const float Alpha     = 0.85f;
    private const float Beta      = 0.30f;
    private const float SizeAlpha = 0.50f;   // smoothing for width/height

    private float _cx, _cy, _w, _h;          // current box estimate (pixels)
    private float _vcx, _vcy;                 // centre velocity (pixels / frame)

    /// <summary>Stable identifier, unique for this track's lifetime.</summary>
    public int Id { get; }
    /// <summary>Total detections matched to this track.</summary>
    public int Hits { get; private set; }
    /// <summary>Frames since the track was created.</summary>
    public int Age { get; private set; }
    /// <summary>Frames since the track was last matched to a detection (0 = matched this frame).</summary>
    public int TimeSinceUpdate { get; private set; }
    /// <summary>Whether the track has enough hits to be reported (set by the tracker).</summary>
    public bool Confirmed { get; internal set; }
    /// <summary>Class label carried over from the matched detections.</summary>
    public string Label { get; private set; }

    // Whether this track has already been added to the tracker's unique tally.
    internal bool CountedUnique;

    public float X1 => _cx - _w * 0.5f;
    public float Y1 => _cy - _h * 0.5f;
    public float X2 => _cx + _w * 0.5f;
    public float Y2 => _cy + _h * 0.5f;
    public float CenterX => _cx;
    public float CenterY => _cy;

    internal Track(int id, in Detection d)
    {
        Id = id;
        _cx = d.CenterX; _cy = d.CenterY; _w = d.Width; _h = d.Height;
        Label = d.Label;
        Hits = 1; Age = 1; TimeSinceUpdate = 0;
    }

    /// <summary>Advance the estimate by one frame using the current velocity.</summary>
    internal void Predict()
    {
        _cx += _vcx;
        _cy += _vcy;
        Age++;
        TimeSinceUpdate++;
    }

    /// <summary>Correct the estimate with a matched detection (alpha-beta update).</summary>
    internal void Update(in Detection d)
    {
        float rx = d.CenterX - _cx;   // residual after prediction
        float ry = d.CenterY - _cy;
        _cx += Alpha * rx;
        _cy += Alpha * ry;
        _vcx += Beta * rx;
        _vcy += Beta * ry;
        _w += SizeAlpha * (d.Width  - _w);
        _h += SizeAlpha * (d.Height - _h);
        Label = d.Label;
        Hits++;
        TimeSinceUpdate = 0;
    }

    /// <summary>The current box estimate as a <see cref="Detection"/> (for IoU matching / drawing).</summary>
    public Detection ToDetection() => new(X1, Y1, X2, Y2, 1f, Label);
}
