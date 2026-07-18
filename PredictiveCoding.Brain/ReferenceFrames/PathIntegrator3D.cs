namespace PredictiveCoding.Brain;

using System.Numerics;

// ─────────────────────────────────────────────────────────────────────────────
// PathIntegrator3D  —  motor-driven location update over SE(3) (grid-cell path
// integration in 3-D).
//
// The SE(3) twin of PathIntegrator. Grid cells maintain a sense of location by
// integrating self-movement: each motor command updates the location code so the
// cortex can predict the feature it will sense next (Hawkins et al. 2019, P3).
// This module tracks a continuous 3-D pose — position as (x, y, z), orientation
// as a unit quaternion so repeated rotations accumulate no Euler round-trip
// drift — and emits the corresponding location SDR (via PoseEncoder3D) after
// every move. Step applies a world-frame displacement (a displacement cell),
// while StepForward/Rotate give the egocentric "move along my forward axis /
// rotate in my own frame" primitives an agent uses.
//
// Because integration is exact in continuous space, a closed motor loop returns
// to the starting location — planar (4 × [forward; yaw 90°]) or fully 3-D
// (climb and descend) — the consistency property path integration provides.
//
// Reference: Hawkins et al. (2019) P3 (grid cells & path integration).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class PathIntegrator3D
{
    private readonly PoseEncoder3D _encoder;
    private Quaternion _orientation;   // unit quaternion, local → world rotation

    /// <summary>Current x coordinate.</summary>
    public float X { get; private set; }

    /// <summary>Current y coordinate.</summary>
    public float Y { get; private set; }

    /// <summary>Current z coordinate.</summary>
    public float Z { get; private set; }

    /// <summary>Current orientation as a unit quaternion (local → world).</summary>
    public Quaternion Orientation => _orientation;

    /// <summary>The current pose as a <see cref="Pose3D"/>.</summary>
    public Pose3D Pose
    {
        get
        {
            var (yaw, pitch, roll) = ToYawPitchRoll(_orientation);
            return Pose3D.FromTranslation(X, Y, Z)
                         .Compose(Pose3D.FromYawPitchRoll(yaw, pitch, roll));
        }
    }

    /// <summary>The location SDR for the current pose (grid-cell code).</summary>
    public Sdr LocationCode
    {
        get
        {
            var (yaw, pitch, roll) = Pose.ToYawPitchRoll();
            return _encoder.Encode(X, Y, Z, yaw, pitch, roll);
        }
    }

    /// <param name="encoder">Encodes (x, y, z, yaw, pitch, roll) to the location SDR.</param>
    public PathIntegrator3D(
        PoseEncoder3D encoder, float x0 = 0f, float y0 = 0f, float z0 = 0f,
        float yaw0 = 0f, float pitch0 = 0f, float roll0 = 0f)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        _encoder = encoder;
        X = x0;
        Y = y0;
        Z = z0;
        _orientation = FromYawPitchRoll(yaw0, pitch0, roll0);
    }

    /// <summary>Integrates a world-frame displacement (dx, dy, dz) and returns the
    /// new location SDR. This is the raw displacement-cell update.</summary>
    public Sdr Step(float dx, float dy, float dz)
    {
        X += dx;
        Y += dy;
        Z += dz;
        return LocationCode;
    }

    /// <summary>Moves <paramref name="distance"/> along the current forward vector
    /// (the orientation applied to +X — egocentric forward motion) and returns the
    /// new location SDR.</summary>
    public Sdr StepForward(float distance)
    {
        Vector3 forward = Vector3.Transform(Vector3.UnitX, _orientation);
        X += distance * forward.X;
        Y += distance * forward.Y;
        Z += distance * forward.Z;
        return LocationCode;
    }

    /// <summary>Rotates egocentrically by (yaw, pitch, roll) degrees — the delta is
    /// applied in the LOCAL frame, the same composition order Pose3D.Compose uses
    /// for the rotation part — and returns the (unchanged-position) location SDR.</summary>
    public Sdr Rotate(float yawDeg, float pitchDeg, float rollDeg)
    {
        _orientation = Quaternion.Normalize(_orientation * FromYawPitchRoll(yawDeg, pitchDeg, rollDeg));
        return LocationCode;
    }

    /// <summary>Resets the pose to the given coordinates and orientation.</summary>
    public void Reset(float x = 0f, float y = 0f, float z = 0f,
                      float yaw = 0f, float pitch = 0f, float roll = 0f)
    {
        X = x;
        Y = y;
        Z = z;
        _orientation = FromYawPitchRoll(yaw, pitch, roll);
    }

    // ── Yaw/pitch/roll ↔ quaternion plumbing ─────────────────────────────────
    //
    // Same convention as Pose3D: yaw about +Z (X toward Y), positive pitch lifts
    // the forward +X axis toward +Z (rotation about −Y), roll about the forward
    // +X axis (Y toward Z); intrinsic order yaw → pitch → roll.

    private static Quaternion FromYawPitchRoll(float yawDeg, float pitchDeg, float rollDeg)
    {
        float yawRad = yawDeg * MathF.PI / 180f;
        float pitchRad = pitchDeg * MathF.PI / 180f;
        float rollRad = rollDeg * MathF.PI / 180f;

        Quaternion qYaw = Quaternion.CreateFromAxisAngle(Vector3.UnitZ, yawRad);
        Quaternion qPitch = Quaternion.CreateFromAxisAngle(-Vector3.UnitY, pitchRad);
        Quaternion qRoll = Quaternion.CreateFromAxisAngle(Vector3.UnitX, rollRad);

        // System.Numerics q1 * q2 applies q2 first, so this is R(yaw)·R(pitch)·R(roll):
        // intrinsic yaw, then pitch, then roll.
        return Quaternion.Normalize(qYaw * qPitch * qRoll);
    }

    private static (float YawDeg, float PitchDeg, float RollDeg) ToYawPitchRoll(Quaternion q)
    {
        q = Quaternion.Normalize(q);

        double sinPitch = Math.Clamp(2.0 * (q.X * q.Z - q.W * q.Y), -1.0, 1.0);
        double pitch = Math.Asin(sinPitch);
        double yaw = Math.Atan2(2.0 * (q.X * q.Y + q.W * q.Z), 1.0 - 2.0 * (q.Y * q.Y + q.Z * q.Z));
        double roll = Math.Atan2(2.0 * (q.Y * q.Z + q.W * q.X), 1.0 - 2.0 * (q.X * q.X + q.Y * q.Y));

        const double toDeg = 180.0 / Math.PI;
        return ((float)(yaw * toDeg), (float)(pitch * toDeg), (float)(roll * toDeg));
    }
}
