namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// PathIntegrator  —  motor-driven location update (grid-cell path integration).
//
// Grid cells maintain a sense of location by integrating self-movement: each motor
// command updates the location code so the cortex can predict the feature it will
// sense next (Hawkins et al. 2019, P3). This module tracks a continuous pose and
// emits the corresponding location SDR (via PoseEncoder) after every move. Step
// applies a world-frame displacement (a displacement cell), while StepForward/Turn
// give the egocentric "move along my heading / rotate" primitives an agent uses.
//
// Because integration is exact in continuous space, a closed motor loop returns to
// the starting location — the consistency property path integration provides.
//
// Reference: Hawkins et al. (2019) P3 (grid cells & path integration).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class PathIntegrator
{
    private readonly PoseEncoder _encoder;

    /// <summary>Current x coordinate.</summary>
    public float X { get; private set; }

    /// <summary>Current y coordinate.</summary>
    public float Y { get; private set; }

    /// <summary>Current heading in degrees.</summary>
    public float HeadingDeg { get; private set; }

    /// <summary>The current pose as a <see cref="Pose2D"/>.</summary>
    public Pose2D Pose => new(X, Y, HeadingDeg);

    /// <summary>The location SDR for the current pose (grid-cell code).</summary>
    public Sdr LocationCode => _encoder.Encode(X, Y, HeadingDeg);

    /// <param name="encoder">Encodes (x, y, θ) to the location SDR.</param>
    public PathIntegrator(PoseEncoder encoder, float x0 = 0f, float y0 = 0f, float heading0 = 0f)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        _encoder = encoder;
        X = x0;
        Y = y0;
        HeadingDeg = Normalize(heading0);
    }

    /// <summary>Integrates a world-frame displacement (dx, dy) and returns the new
    /// location SDR. This is the raw displacement-cell update.</summary>
    public Sdr Step(float dx, float dy)
    {
        X += dx;
        Y += dy;
        return LocationCode;
    }

    /// <summary>Moves <paramref name="distance"/> along the current heading (egocentric
    /// forward motion) and returns the new location SDR.</summary>
    public Sdr StepForward(float distance)
    {
        double rad = HeadingDeg * Math.PI / 180.0;
        X += (float)(distance * Math.Cos(rad));
        Y += (float)(distance * Math.Sin(rad));
        return LocationCode;
    }

    /// <summary>Rotates the heading by <paramref name="deltaDeg"/> and returns the
    /// (unchanged-position) location SDR.</summary>
    public Sdr Turn(float deltaDeg)
    {
        HeadingDeg = Normalize(HeadingDeg + deltaDeg);
        return LocationCode;
    }

    /// <summary>Resets the pose to the given coordinates and heading.</summary>
    public void Reset(float x = 0f, float y = 0f, float heading = 0f)
    {
        X = x;
        Y = y;
        HeadingDeg = Normalize(heading);
    }

    private static float Normalize(float deg) => ((deg % 360f) + 360f) % 360f;
}
