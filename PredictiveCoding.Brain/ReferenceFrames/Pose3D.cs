using System.Numerics;

namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// Pose3D  —  continuous 3-D rigid-body pose / relative transform (x, y, z, q).
//
// The SE(3) twin of Pose2D: where Pose2D carries the planar rigid transform a
// displacement cell represents — the transform relating one reference frame to
// another (Hawkins et al. 2019, P3 "displacement cells") — Pose3D carries the
// full spatial one, with translation (X, Y, Z) and rotation as a unit
// quaternion. Compose nests frames exactly as in 2-D:
// parent.Compose(childInParent) yields the child frame expressed in the
// parent's parent — the core operation behind objects composed of objects.
//
// Rotation convention (pinned by the anchor cases below; a yaw-only pose with
// z = 0 must match Pose2D exactly):
//   • yaw    rotates about +Z, taking +X toward +Y (the Pose2D θ);
//   • pitch  rotates about −Y, so positive pitch lifts the forward +X axis
//     toward +Z;
//   • roll   rotates about the forward +X axis, taking +Y toward +Z;
//   • FromYawPitchRoll applies them intrinsically: yaw first, then pitch,
//     then roll.
// Anchor cases:
//   A1  FromYawPitchRoll(90,0,0).Apply(1,0,0) ≈ (0,1,0)
//   A2  FromYawPitchRoll(0,90,0).Apply(1,0,0) ≈ (0,0,1)
//   A3  FromYawPitchRoll(0,0,90).Apply(0,1,0) ≈ (0,0,1)
//   A4  FromYawPitchRoll(y,p,r) == yawOnly.Compose(pitchOnly).Compose(rollOnly)
//   A5  ToYawPitchRoll round-trips FromYawPitchRoll for pitch in (−90, 90)
//
// Reference: Hawkins et al. (2019) P3 (displacement cells); Leadholm et al.
// (2025) P9 §3 (relative pose between sub-objects).
// ─────────────────────────────────────────────────────────────────────────────

public readonly record struct Pose3D(float X, float Y, float Z, Quaternion Rotation)
{
    /// <summary>The zero transform (no translation, no rotation).</summary>
    public static Pose3D Identity => new(0f, 0f, 0f, Quaternion.Identity);

    /// <summary>The translation component as a displacement vector.</summary>
    public (float Dx, float Dy, float Dz) Translation => (X, Y, Z);

    /// <summary>A pure translation (no rotation).</summary>
    public static Pose3D FromTranslation(float dx, float dy, float dz)
        => new(dx, dy, dz, Quaternion.Identity);

    /// <summary>A pure rotation (no translation) from intrinsic yaw → pitch → roll
    /// angles in degrees: yaw about +Z (+X toward +Y, the Pose2D θ), then pitch
    /// about −Y (positive pitch lifts +X toward +Z), then roll about the forward
    /// +X axis (+Y toward +Z).</summary>
    public static Pose3D FromYawPitchRoll(float yawDeg, float pitchDeg, float rollDeg)
    {
        var qYaw = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, ToRad(yawDeg));
        var qPitch = Quaternion.CreateFromAxisAngle(-Vector3.UnitY, ToRad(pitchDeg));
        var qRoll = Quaternion.CreateFromAxisAngle(Vector3.UnitX, ToRad(rollDeg));

        // System.Numerics' q1 * q2 applies q2 to a vector first, then q1 — so
        // yaw * pitch * roll rotates by roll, then pitch, then yaw about the
        // fixed axes, which is exactly intrinsic yaw → pitch → roll (anchor A4).
        return new Pose3D(0f, 0f, 0f, qYaw * qPitch * qRoll);
    }

    /// <summary>Maps a point from this frame's local coordinates into its parent
    /// frame: rotate by the quaternion, then translate by (X, Y, Z).</summary>
    public (float X, float Y, float Z) Apply(float px, float py, float pz)
    {
        var r = Vector3.Transform(new Vector3(px, py, pz), Rotation);
        return (X + r.X, Y + r.Y, Z + r.Z);
    }

    /// <summary>Composes this pose with a <paramref name="child"/> pose expressed in
    /// this frame, returning the child frame expressed in this pose's parent frame.
    /// Translations compose through <see cref="Apply"/>; rotations compose with the
    /// child's rotation applied first.</summary>
    public Pose3D Compose(Pose3D child)
    {
        var (cx, cy, cz) = Apply(child.X, child.Y, child.Z);
        return new Pose3D(cx, cy, cz, Rotation * child.Rotation);
    }

    /// <summary>The inverse transform: <c>p.Compose(p.Inverse()) == Identity</c>
    /// (up to floating-point error).</summary>
    public Pose3D Inverse()
    {
        var invRotation = Quaternion.Inverse(Rotation);
        var it = Vector3.Transform(new Vector3(-X, -Y, -Z), invRotation);
        return new Pose3D(it.X, it.Y, it.Z, invRotation);
    }

    /// <summary>This pose with the quaternion normalised to unit length (rotation
    /// quaternions drift off the unit sphere under repeated composition).</summary>
    public Pose3D Normalized() => this with { Rotation = Quaternion.Normalize(Rotation) };

    /// <summary>Decomposes the rotation back into intrinsic yaw → pitch → roll
    /// angles in degrees, with yaw and roll in (−180, 180] and pitch in [−90, 90].
    /// Round-trips <see cref="FromYawPitchRoll"/> for pitch in (−90, 90). At
    /// gimbal lock (pitch = ±90°) yaw and roll rotate about the same world axis
    /// and only their combination is observable; this method reports the
    /// combination as yaw and returns roll = 0.</summary>
    public (float YawDeg, float PitchDeg, float RollDeg) ToYawPitchRoll()
    {
        var q = Quaternion.Normalize(Rotation);
        float x = q.X, y = q.Y, z = q.Z, w = q.W;

        // Rotation-matrix elements of the yaw(+Z) → pitch(−Y) → roll(+X)
        // factorisation; sinPitch is the matrix entry R31 = 2(xz − wy).
        float sinPitch = Math.Clamp(2f * (x * z - w * y), -1f, 1f);

        if (Math.Abs(sinPitch) > 0.99999f)   // gimbal lock: fold roll into yaw
        {
            float yawLocked = MathF.Atan2(2f * (w * z - x * y), 1f - 2f * (x * x + z * z));
            return (ToDeg(yawLocked), MathF.CopySign(90f, sinPitch), 0f);
        }

        float yaw = MathF.Atan2(2f * (x * y + w * z), 1f - 2f * (y * y + z * z));
        float pitch = MathF.Asin(sinPitch);
        float roll = MathF.Atan2(2f * (y * z + w * x), 1f - 2f * (x * x + y * y));
        return (ToDeg(yaw), ToDeg(pitch), ToDeg(roll));
    }

    private static float ToRad(float deg) => deg * (MathF.PI / 180f);

    private static float ToDeg(float rad) => rad * (180f / MathF.PI);
}
