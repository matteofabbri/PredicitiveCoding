namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// FeaturePoint3D  —  a sensed feature at a continuous location in a 3-D
// reference frame.
//
// The SE(3) twin of FeaturePoint: (X, Y, Z) is the position in the object's own
// frame and Feature is the SDR sensed there. Keeping the location continuous
// lets compositional transforms (Pose3D) translate and rotate whole sub-objects
// before the points are encoded to SDRs for a CorticalColumn.
//
// Reference: Hawkins et al. (2019) (features anchored in object-centric frames).
// ─────────────────────────────────────────────────────────────────────────────

public readonly record struct FeaturePoint3D(float X, float Y, float Z, Sdr Feature)
{
    /// <summary>Returns this point with its location mapped through
    /// <paramref name="pose"/> (the feature SDR is unchanged).</summary>
    public FeaturePoint3D Transformed(Pose3D pose)
    {
        var (x, y, z) = pose.Apply(X, Y, Z);
        return new FeaturePoint3D(x, y, z, Feature);
    }
}
