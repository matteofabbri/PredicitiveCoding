using PredictiveCoding.Core;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// Hypothalamus  —  homeostatic set-points / drives
//
// The hypothalamus holds the body's regulated set-points (temperature, glucose,
// osmolarity, …) and generates a drive whenever the internal state departs from
// them.  In active inference these set-points are prior preferences: the drive is
// the error (set-point − current) and the set-point itself is the goal observation
// a controller (e.g. MotorCortex) acts to bring about.  A pure controller, no
// persistence.
//
// References: Sterling (2012) allostasis; Pezzulo, Rigoli & Friston (2015) drives.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class Hypothalamus
{
    private readonly float[] _setPoints;

    /// <summary>The regulated set-points (prior preferences).</summary>
    public IReadOnlyList<float> SetPoints => _setPoints;

    /// <summary>Number of regulated variables.</summary>
    public int Dim => _setPoints.Length;

    public Hypothalamus(IReadOnlyList<float> setPoints)
    {
        if (setPoints is null || setPoints.Count == 0)
            throw new ArgumentException("At least one set-point is required.", nameof(setPoints));
        _setPoints = setPoints.ToArray();
    }

    /// <summary>Per-variable homeostatic error (set-point − current): the drive signal.</summary>
    public float[] Drive(IReadOnlyList<float> current)
    {
        if (current.Count != Dim)
            throw new ArgumentException($"current has {current.Count} values; expected {Dim}.");
        var d = new float[Dim];
        for (int i = 0; i < Dim; i++) d[i] = _setPoints[i] - current[i];
        return d;
    }

    /// <summary>Overall need = L2 norm of the drive (0 when exactly at the set-point).</summary>
    public float NeedLevel(IReadOnlyList<float> current)
    {
        var d = Drive(current);
        float s = 0f;
        foreach (var v in d) s += v * v;
        return MathF.Sqrt(s);
    }

    /// <summary>The homeostatic goal observation (the set-points) for a controller such as MotorCortex.</summary>
    public DenseTensor<float> Goal() => new(_setPoints.ToArray(), Dim);
}
