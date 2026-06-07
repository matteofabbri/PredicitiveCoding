namespace PredictiveCoding.Vision.ReId;

/// <summary>
/// A register of distinct people by appearance embedding. New embeddings are matched
/// against the gallery by cosine similarity: a match at or above the threshold reuses
/// that person's stable id, otherwise a new person is registered. This is the dedup
/// that turns per-track identities into a true "how many distinct people have I seen"
/// count, even when the tracker hands the same person a fresh id after an occlusion or
/// re-entry. Mirrors the cosine register/identify pattern of the face FaceDatabase.
/// </summary>
public sealed class PersonGallery
{
    private readonly List<(int Id, float[] Embedding)> _people = [];
    private readonly double _matchThreshold;
    private int _nextId = 1;

    /// <summary>Number of distinct people registered.</summary>
    public int Count => _people.Count;

    /// <param name="matchThreshold">Cosine similarity in [0,1] at/above which two
    /// embeddings are treated as the same person. Default 0.6.</param>
    public PersonGallery(double matchThreshold = 0.6)
    {
        if (matchThreshold is < 0 or > 1)
            throw new ArgumentOutOfRangeException(nameof(matchThreshold));
        _matchThreshold = matchThreshold;
    }

    /// <summary>Best matching person id and score for <paramref name="embedding"/>,
    /// or null when the gallery is empty.</summary>
    public (int Id, double Score)? Identify(float[] embedding)
    {
        ArgumentNullException.ThrowIfNull(embedding);
        if (_people.Count == 0) return null;

        int bestId = _people[0].Id;
        double bestScore = double.MinValue;
        foreach (var (id, emb) in _people)
        {
            double s = Cosine(embedding, emb);
            if (s > bestScore) { bestScore = s; bestId = id; }
        }
        return (bestId, bestScore);
    }

    /// <summary>Returns the id of the matching person if one is within the threshold,
    /// otherwise registers <paramref name="embedding"/> as a new person and returns the
    /// new id.</summary>
    public int IdentifyOrRegister(float[] embedding)
    {
        ArgumentNullException.ThrowIfNull(embedding);
        if (Identify(embedding) is { } m && m.Score >= _matchThreshold)
            return m.Id;

        int id = _nextId++;
        _people.Add((id, (float[])embedding.Clone()));
        return id;
    }

    /// <summary>Clears all registered people.</summary>
    public void Reset()
    {
        _people.Clear();
        _nextId = 1;
    }

    private static double Cosine(float[] a, float[] b)
    {
        int n = Math.Min(a.Length, b.Length);
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < n; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        double denom = Math.Sqrt(na) * Math.Sqrt(nb);
        return denom > 1e-12 ? dot / denom : 0.0;
    }
}
