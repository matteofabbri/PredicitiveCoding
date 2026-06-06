namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// FeaturePoint  —  a sensed feature at a continuous location in a reference frame.
//
// The continuous-space analogue of an ObjectFrame's (location, feature) pair:
// (X, Y) is the position in the object's own frame and Feature is the SDR sensed
// there. Keeping the location continuous lets compositional transforms (Pose2D)
// translate and rotate whole sub-objects before the points are encoded to SDRs
// for a CorticalColumn.
//
// Reference: Hawkins et al. (2019) (features anchored in object-centric frames).
// ─────────────────────────────────────────────────────────────────────────────

public readonly record struct FeaturePoint(float X, float Y, Sdr Feature)
{
    /// <summary>Returns this point with its location mapped through
    /// <paramref name="pose"/> (the feature SDR is unchanged).</summary>
    public FeaturePoint Transformed(Pose2D pose)
    {
        var (x, y) = pose.Apply(X, Y);
        return new FeaturePoint(x, y, Feature);
    }
}
