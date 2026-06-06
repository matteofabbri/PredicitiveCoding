namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// Pose2D  —  continuous 2-D rigid-body pose / relative transform (x, y, θ).
//
// The value-type counterpart of DisplacementModule (translation) and PoseEncoder
// (orientation): where those produce SDRs, Pose2D carries the underlying
// continuous relative pose and the algebra to compose it. A Pose2D is exactly
// what a displacement cell represents — the rigid transform that relates one
// reference frame to another (Hawkins et al. 2019, "displacement cells").
//
// Compose nests frames: parent.Compose(childInParent) yields the child frame
// expressed in the parent's parent — the core operation behind objects composed
// of objects (Phase 8).
//
// Reference: Hawkins et al. (2019) Fig. 4 (displacement cells);
// Leadholm et al. (2025) Section 3 (relative pose between sub-objects).
// ─────────────────────────────────────────────────────────────────────────────

public readonly record struct Pose2D(float X, float Y, float ThetaDeg)
{
    /// <summary>The zero transform (no translation, no rotation).</summary>
    public static Pose2D Identity => new(0f, 0f, 0f);

    /// <summary>The translation component as a displacement vector.</summary>
    public (float Dx, float Dy) Translation => (X, Y);

    /// <summary>A pure translation (no rotation).</summary>
    public static Pose2D FromTranslation(float dx, float dy) => new(dx, dy, 0f);

    /// <summary>Maps a point from this frame's local coordinates into its parent
    /// frame: rotate by θ, then translate by (X, Y).</summary>
    public (float X, float Y) Apply(float px, float py)
    {
        var (rx, ry) = Rotate(px, py, ThetaDeg);
        return (X + rx, Y + ry);
    }

    /// <summary>Composes this pose with a <paramref name="child"/> pose expressed in
    /// this frame, returning the child frame expressed in this pose's parent frame.
    /// Translations compose through <see cref="Apply"/>; rotations add.</summary>
    public Pose2D Compose(Pose2D child)
    {
        var (cx, cy) = Apply(child.X, child.Y);
        return new Pose2D(cx, cy, ThetaDeg + child.ThetaDeg);
    }

    /// <summary>The inverse transform: <c>p.Compose(p.Inverse()) == Identity</c>
    /// (up to floating-point error).</summary>
    public Pose2D Inverse()
    {
        var (ix, iy) = Rotate(-X, -Y, -ThetaDeg);
        return new Pose2D(ix, iy, -ThetaDeg);
    }

    /// <summary>This pose with θ wrapped into <c>[0, 360)</c>.</summary>
    public Pose2D Normalized() => this with { ThetaDeg = ((ThetaDeg % 360f) + 360f) % 360f };

    private static (float X, float Y) Rotate(float px, float py, float thetaDeg)
    {
        double rad = thetaDeg * Math.PI / 180.0;
        double cos = Math.Cos(rad), sin = Math.Sin(rad);
        return ((float)(px * cos - py * sin), (float)(px * sin + py * cos));
    }
}
