namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// DisplacementRecognizer3D  —  pose-invariant object recognition via displacement
// structure, in 3-D (the SE(3) twin of DisplacementRecognizer).
//
// The CorticalColumn / ObjectMemory recognises objects by matching *absolute*
// (location, feature) pairs, so a rigidly moved object (rotated and/or translated)
// no longer matches — the limitation noted at the end of Phase 9. Displacement
// cells (Hawkins et al. 2019, P3) solve this: the relative pose between two sensed
// features does not depend on where the object sits in the world. The pairwise
// Euclidean *distance* over (X, Y, Z) between two landmarks is invariant under the
// full SE(3) group of rigid motions (3-D translation + rotation) — which is exactly
// why the 2-D signature carries over unchanged: an object's multiset of
// (featureₐ, feature_b, distance) descriptors is a pose-invariant signature.
//
// Learn() stores that signature per object; Rank()/Recognize() vote by counting how
// many observed descriptors match a stored object's, within feature-overlap and
// distance tolerances — robust to translation, rotation, occlusion (a subset of
// pairs) and sensor noise (tolerant feature/distance matching).
//
// Reference: Hawkins et al. (2019) P3 (displacement cells); Leadholm et al. (2025)
// P9 §3 (relative pose between sub-features).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class DisplacementRecognizer3D
{
    private readonly int _minOverlap;
    private readonly float _distanceTolerance;
    private readonly Dictionary<int, IReadOnlyList<PairDescriptor>> _objects = [];

    /// <summary>Number of learned objects.</summary>
    public int ObjectCount => _objects.Count;

    /// <param name="minOverlap">Minimum SDR overlap for two features to be considered the same.</param>
    /// <param name="distanceTolerance">Maximum absolute difference between two pairwise
    /// distances for them to count as equal (in the same units as the landmark coordinates).</param>
    public DisplacementRecognizer3D(int minOverlap = 12, float distanceTolerance = 0.6f)
    {
        if (minOverlap < 1) throw new ArgumentOutOfRangeException(nameof(minOverlap));
        if (distanceTolerance < 0f) throw new ArgumentOutOfRangeException(nameof(distanceTolerance));
        _minOverlap = minOverlap;
        _distanceTolerance = distanceTolerance;
    }

    /// <summary>Learns (or replaces) an object from its landmarks, stored as a
    /// pose-invariant set of pairwise displacement descriptors.</summary>
    public void Learn(int objectId, IReadOnlyList<FeaturePoint3D> landmarks)
    {
        ArgumentNullException.ThrowIfNull(landmarks);
        _objects[objectId] = BuildDescriptors(landmarks);
    }

    /// <summary>Ranks every learned object by how well its displacement structure
    /// matches <paramref name="observed"/>, best first.</summary>
    public IReadOnlyList<RecognitionScore> Rank(IReadOnlyList<FeaturePoint3D> observed)
    {
        ArgumentNullException.ThrowIfNull(observed);
        var obs = BuildDescriptors(observed);

        var scores = new List<RecognitionScore>(_objects.Count);
        foreach (var (id, stored) in _objects)
        {
            int matches = CountMatches(obs, stored);
            double confidence = obs.Count > 0 ? (double)matches / obs.Count : 0.0;
            scores.Add(new RecognitionScore(id, matches, confidence));
        }

        scores.Sort((a, b) => b.MatchingPairs.CompareTo(a.MatchingPairs));
        return scores;
    }

    /// <summary>The best-matching object id, or null if nothing matches at all.</summary>
    public int? Recognize(IReadOnlyList<FeaturePoint3D> observed)
    {
        var ranked = Rank(observed);
        return ranked.Count > 0 && ranked[0].MatchingPairs > 0 ? ranked[0].ObjectId : null;
    }

    // ── Internals ─────────────────────────────────────────────────────────────

    private readonly record struct PairDescriptor(Sdr FeatureA, Sdr FeatureB, float Distance);

    private static IReadOnlyList<PairDescriptor> BuildDescriptors(IReadOnlyList<FeaturePoint3D> landmarks)
    {
        var descriptors = new List<PairDescriptor>(landmarks.Count * (landmarks.Count - 1) / 2);
        for (int i = 0; i < landmarks.Count; i++)
            for (int j = i + 1; j < landmarks.Count; j++)
                descriptors.Add(new PairDescriptor(
                    landmarks[i].Feature, landmarks[j].Feature, Distance(landmarks[i], landmarks[j])));
        return descriptors;
    }

    private int CountMatches(IReadOnlyList<PairDescriptor> observed, IReadOnlyList<PairDescriptor> stored)
    {
        int matches = 0;
        foreach (var o in observed)
            foreach (var s in stored)
                if (Math.Abs(o.Distance - s.Distance) <= _distanceTolerance && FeaturesMatch(o, s))
                {
                    matches++;
                    break;   // each observed descriptor votes at most once
                }
        return matches;
    }

    // A pair matches in either orientation: {A,B} is unordered.
    private bool FeaturesMatch(PairDescriptor o, PairDescriptor s)
        => (o.FeatureA.Matches(s.FeatureA, _minOverlap) && o.FeatureB.Matches(s.FeatureB, _minOverlap))
        || (o.FeatureA.Matches(s.FeatureB, _minOverlap) && o.FeatureB.Matches(s.FeatureA, _minOverlap));

    private static float Distance(FeaturePoint3D a, FeaturePoint3D b)
    {
        float dx = a.X - b.X, dy = a.Y - b.Y, dz = a.Z - b.Z;
        return MathF.Sqrt(dx * dx + dy * dy + dz * dz);
    }
}
