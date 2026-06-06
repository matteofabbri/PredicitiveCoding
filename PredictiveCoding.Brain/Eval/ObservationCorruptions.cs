namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// ObservationCorruptions  —  the test conditions of the recognition benchmark.
//
// Mirrors the robustness axes the Thousand-Brains papers evaluate against
// (Leadholm et al. 2025): partial views (occlusion), noisy sensors (feature
// corruption), and imprecise pose (location jitter). Each transform takes a clean
// list of FeaturePoints and returns a corrupted copy, so a benchmark can score the
// same learned models under every condition.
//
// Reference: Leadholm et al. (2025) P9; Clay et al. (2024) P2
// (rapid, robust recognition under novel poses, occlusion and sensor noise).
// ─────────────────────────────────────────────────────────────────────────────

public static class ObservationCorruptions
{
    /// <summary>Occlusion: keeps a random <paramref name="keepFraction"/> of the points
    /// (at least one), simulating a partial view of the object.</summary>
    public static IReadOnlyList<FeaturePoint> Occlude(
        IReadOnlyList<FeaturePoint> points, float keepFraction, Random rng)
    {
        ArgumentNullException.ThrowIfNull(points);
        ArgumentNullException.ThrowIfNull(rng);
        if (keepFraction is <= 0f or > 1f)
            throw new ArgumentOutOfRangeException(nameof(keepFraction), "Must be in (0, 1].");

        int keep = Math.Max(1, (int)Math.Round(keepFraction * points.Count));
        return Shuffled(points, rng).Take(keep).ToList();
    }

    /// <summary>Sensor noise: flips a <paramref name="noiseFraction"/> of each feature
    /// SDR's active bits to random inactive positions (cardinality preserved). Locations
    /// are untouched.</summary>
    public static IReadOnlyList<FeaturePoint> AddSensorNoise(
        IReadOnlyList<FeaturePoint> points, float noiseFraction, Random rng)
    {
        ArgumentNullException.ThrowIfNull(points);
        ArgumentNullException.ThrowIfNull(rng);

        var result = new List<FeaturePoint>(points.Count);
        foreach (var pt in points)
            result.Add(pt with { Feature = pt.Feature.AddNoise(noiseFraction, rng) });
        return result;
    }

    /// <summary>Pose jitter: perturbs each point's location by a uniform offset in
    /// <c>[−maxJitter, +maxJitter]</c> on each axis, simulating imprecise sensor pose.
    /// Features are untouched.</summary>
    public static IReadOnlyList<FeaturePoint> JitterPose(
        IReadOnlyList<FeaturePoint> points, float maxJitter, Random rng)
    {
        ArgumentNullException.ThrowIfNull(points);
        ArgumentNullException.ThrowIfNull(rng);
        if (maxJitter < 0f) throw new ArgumentOutOfRangeException(nameof(maxJitter));

        var result = new List<FeaturePoint>(points.Count);
        foreach (var pt in points)
        {
            float dx = (float)(rng.NextDouble() * 2 - 1) * maxJitter;
            float dy = (float)(rng.NextDouble() * 2 - 1) * maxJitter;
            result.Add(pt with { X = pt.X + dx, Y = pt.Y + dy });
        }
        return result;
    }

    private static List<FeaturePoint> Shuffled(IReadOnlyList<FeaturePoint> points, Random rng)
    {
        var copy = points.ToList();
        for (int i = copy.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (copy[i], copy[j]) = (copy[j], copy[i]);
        }
        return copy;
    }
}
