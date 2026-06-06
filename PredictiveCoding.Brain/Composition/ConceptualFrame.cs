namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// ConceptualFrame  —  an abstract, N-dimensional reference frame.
//
// The Thousand Brains theory holds that the cortex uses the same grid-cell /
// reference-frame machinery for abstract concepts as for physical space: ideas are
// arranged at locations in a conceptual space and "movement" through that space is
// reasoning (Hawkins 2021, ch. 8). ConceptualFrame generalises the physical
// PoseEncoder to an arbitrary number of axes with arbitrary meaning (a flavour
// space, a feature space, a semantic map): each axis is a ScalarEncoder, locations
// are concatenated SDRs, and Move applies a displacement — the abstract analogue of
// path integration — to navigate from one concept toward another.
//
// Reference: Hawkins (2021) ch. 8 (reference frames for concepts / thought).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class ConceptualFrame
{
    private readonly ScalarEncoder[] _axes;
    private readonly List<(double[] Coords, Sdr Concept)> _concepts = [];

    /// <summary>Number of conceptual dimensions (axes).</summary>
    public int Dimensions => _axes.Length;

    /// <summary>Number of concepts anchored in the frame.</summary>
    public int ConceptCount => _concepts.Count;

    /// <summary>Size of a location SDR (dimensions × bitsPerAxis).</summary>
    public int LocationSize => _axes.Sum(a => a.Size);

    /// <param name="axes">Per-axis (min, max) ranges; one entry per conceptual dimension.</param>
    /// <param name="bitsPerAxis">Total SDR bits per axis. Default 100.</param>
    /// <param name="activePerAxis">Active bits per axis. Default 10.</param>
    public ConceptualFrame(IReadOnlyList<(double Min, double Max)> axes, int bitsPerAxis = 100, int activePerAxis = 10)
    {
        ArgumentNullException.ThrowIfNull(axes);
        if (axes.Count < 1) throw new ArgumentException("At least one axis is required.", nameof(axes));
        _axes = new ScalarEncoder[axes.Count];
        for (int i = 0; i < axes.Count; i++)
            _axes[i] = new ScalarEncoder(axes[i].Min, axes[i].Max, bitsPerAxis, activePerAxis);
    }

    /// <summary>Encodes a point in concept space as a concatenated location SDR.
    /// Nearby concepts share active bits; distant ones do not.</summary>
    public Sdr EncodeLocation(IReadOnlyList<double> coords)
    {
        var c = AsVector(coords);
        var bits = new List<int>();
        int offset = 0;
        for (int i = 0; i < _axes.Length; i++)
        {
            foreach (int b in _axes[i].Encode(c[i]).ActiveBits) bits.Add(b + offset);
            offset += _axes[i].Size;
        }
        return new Sdr(LocationSize, bits);
    }

    /// <summary>Anchors <paramref name="concept"/> at <paramref name="coords"/> in the frame.</summary>
    public void AddConcept(IReadOnlyList<double> coords, Sdr concept)
    {
        ArgumentNullException.ThrowIfNull(concept);
        _concepts.Add((AsVector(coords), concept));
    }

    /// <summary>The anchored concept nearest <paramref name="coords"/> by Euclidean
    /// distance in concept space, with its coordinates and distance. Null when the
    /// frame is empty.</summary>
    public (double[] Coords, Sdr Concept, double Distance)? Nearest(IReadOnlyList<double> coords)
    {
        if (_concepts.Count == 0) return null;
        var q = AsVector(coords);

        int best = 0;
        double bestDist = double.PositiveInfinity;
        for (int i = 0; i < _concepts.Count; i++)
        {
            double d = Distance(q, _concepts[i].Coords);
            if (d < bestDist) { bestDist = d; best = i; }
        }
        var (c, concept) = _concepts[best];
        return (c, concept, bestDist);
    }

    /// <summary>The concept nearest <paramref name="coords"/>, or null when the frame
    /// is empty. Convenience wrapper over <see cref="Nearest"/>.</summary>
    public Sdr? Recall(IReadOnlyList<double> coords) => Nearest(coords)?.Concept;

    /// <summary>Applies a <paramref name="displacement"/> to <paramref name="from"/> —
    /// movement through concept space — and returns the destination coordinates.
    /// The abstract analogue of path integration.</summary>
    public double[] Move(IReadOnlyList<double> from, IReadOnlyList<double> displacement)
    {
        var f = AsVector(from);
        var d = AsVector(displacement);
        var result = new double[Dimensions];
        for (int i = 0; i < Dimensions; i++) result[i] = f[i] + d[i];
        return result;
    }

    private double[] AsVector(IReadOnlyList<double> coords)
    {
        ArgumentNullException.ThrowIfNull(coords);
        if (coords.Count != Dimensions)
            throw new ArgumentException(
                $"Expected {Dimensions} coordinates, got {coords.Count}.", nameof(coords));
        return [.. coords];
    }

    private static double Distance(double[] a, double[] b)
    {
        double sum = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double diff = a[i] - b[i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }
}
