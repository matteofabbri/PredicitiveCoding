namespace PredictiveCoding.FaceRecognition.Benchmark;

/// <summary>
/// Flat cosine-similarity gallery for closed-set face identification.
/// Stores one embedding per gallery entry; lookups are O(N) linear scan.
/// </summary>
public sealed class FaceGallery
{
    private readonly List<(string Id, float[] Embedding)> _entries = [];

    public int Count => _entries.Count;

    public void Add(string id, float[] embedding) => _entries.Add((id, embedding));

    /// <summary>Returns the identity with the highest cosine similarity to <paramref name="query"/>.</summary>
    public (string Id, double Score) FindNearest(float[] query)
    {
        string bestId    = "";
        double bestScore = double.NegativeInfinity;
        foreach (var (id, emb) in _entries)
        {
            double s = FaceMath.Cosine(query, emb);
            if (s > bestScore) { bestScore = s; bestId = id; }
        }
        return (bestId, bestScore);
    }

    /// <summary>Returns the top-<paramref name="k"/> identities ordered by cosine similarity.</summary>
    public IReadOnlyList<(string Id, double Score)> FindTopK(float[] query, int k)
        => _entries
            .Select(e => (e.Id, Score: FaceMath.Cosine(query, e.Embedding)))
            .OrderByDescending(x => x.Score)
            .Take(k)
            .ToList();
}
