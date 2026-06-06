namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// DisplacementRecognizer  —  pose-invariant object recognition via displacement
// structure.
//
// The CorticalColumn / ObjectMemory recognises objects by matching *absolute*
// (location, feature) pairs, so a rigidly moved object (rotated and/or translated)
// no longer matches — the limitation noted at the end of Phase 9. Displacement
// cells (Hawkins et al. 2019, P3) solve this: the relative pose between two sensed
// features does not depend on where the object sits in the world. The pairwise
// *distance* between two landmarks is invariant to the full group of rigid motions
// (translation + rotation), so an object's multiset of (featureₐ, feature_b,
// distance) descriptors is a pose-invariant signature.
//
// Learn() stores that signature per object; Rank()/Recognize() vote by counting how
// many observed descriptors match a stored object's, within feature-overlap and
// distance tolerances — robust to translation, rotation, occlusion (a subset of
// pairs) and sensor noise (tolerant feature/distance matching).
//
// Reference: Hawkins et al. (2019) P3 (displacement cells); Leadholm et al. (2025)
// P9 §3 (relative pose between sub-features).
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>How strongly an object matches an observation: the number of pairwise
/// descriptors that agree, and a confidence in [0,1] (matched ÷ observed pairs).</summary>
public sealed record RecognitionScore(int ObjectId, int MatchingPairs, double Confidence);

public sealed class DisplacementRecognizer
{
    private readonly int _minOverlap;
    private readonly float _distanceTolerance;
    private readonly Dictionary<int, IReadOnlyList<PairDescriptor>> _objects = [];

    /// <summary>Number of learned objects.</summary>
    public int ObjectCount => _objects.Count;

    /// <param name="minOverlap">Minimum SDR overlap for two features to be considered the same.</param>
    /// <param name="distanceTolerance">Maximum absolute difference between two pairwise
    /// distances for them to count as equal (in the same units as the landmark coordinates).</param>
    public DisplacementRecognizer(int minOverlap = 12, float distanceTolerance = 0.6f)
    {
        if (minOverlap < 1) throw new ArgumentOutOfRangeException(nameof(minOverlap));
        if (distanceTolerance < 0f) throw new ArgumentOutOfRangeException(nameof(distanceTolerance));
        _minOverlap = minOverlap;
        _distanceTolerance = distanceTolerance;
    }

    /// <summary>Learns (or replaces) an object from its landmarks, stored as a
    /// pose-invariant set of pairwise displacement descriptors.</summary>
    public void Learn(int objectId, IReadOnlyList<FeaturePoint> landmarks)
    {
        ArgumentNullException.ThrowIfNull(landmarks);
        _objects[objectId] = BuildDescriptors(landmarks);
    }

    /// <summary>Ranks every learned object by how well its displacement structure
    /// matches <paramref name="observed"/>, best first.</summary>
    public IReadOnlyList<RecognitionScore> Rank(IReadOnlyList<FeaturePoint> observed)
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
    public int? Recognize(IReadOnlyList<FeaturePoint> observed)
    {
        var ranked = Rank(observed);
        return ranked.Count > 0 && ranked[0].MatchingPairs > 0 ? ranked[0].ObjectId : null;
    }

    // ── Internals ─────────────────────────────────────────────────────────────

    private readonly record struct PairDescriptor(Sdr FeatureA, Sdr FeatureB, float Distance);

    private static IReadOnlyList<PairDescriptor> BuildDescriptors(IReadOnlyList<FeaturePoint> landmarks)
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

    private static float Distance(FeaturePoint a, FeaturePoint b)
    {
        float dx = a.X - b.X, dy = a.Y - b.Y;
        return MathF.Sqrt(dx * dx + dy * dy);
    }
}
