using THEQULT.Indexes;

namespace PredictiveCoding.Benchmark;

/// <summary>
/// 1:N identification (CASIA / VGGFace2 / CelebA / DigiFace / Glint360K style): each probe is matched against a
/// gallery via the THEQULT <see cref="VectorIndex"/>; reports rank-1 and rank-5 (CMC) accuracy.
/// </summary>
public static class IdentificationMetrics
{
    public sealed record Result(double Rank1, double Rank5, int Probes, int GallerySize, int Identities)
    {
        public override string ToString()
            => $"rank-1 {Rank1:P2} · rank-5 {Rank5:P2} · {Probes} probes vs {GallerySize} gallery ({Identities} identities)";
    }

    public static Result Evaluate(
        IReadOnlyList<(string Id, float[] Vector)> gallery,
        IReadOnlyList<(string Id, float[] Vector)> probes,
        int dimension)
    {
        ArgumentNullException.ThrowIfNull(gallery);
        ArgumentNullException.ThrowIfNull(probes);
        if (gallery.Count == 0 || probes.Count == 0)
            return new Result(0, 0, probes.Count, gallery.Count, 0);

        using var store = new InMemoryByteStore();
        using var index = new VectorIndex(
            new VectorIndexOptions { Directory = "in-memory", Dimension = dimension, Metric = VectorMetric.Cosine },
            store);

        for (int i = 0; i < gallery.Count; i++)
            index.Add(gallery[i].Vector, ObjectRef.FromInt64(i));

        int rank1 = 0, rank5 = 0;
        foreach ((string id, float[] vector) in probes)
        {
            IReadOnlyList<VectorMatch> hits = index.Search(vector, 5);
            string[] rankedIds = hits.Select(h => gallery[(int)BitConverter.ToInt64(h.Reference.Span)].Id).ToArray();
            if (rankedIds.Length > 0 && rankedIds[0] == id)
                rank1++;
            if (rankedIds.Contains(id))
                rank5++;
        }

        int identities = gallery.Select(g => g.Id).Distinct().Count();
        return new Result((double)rank1 / probes.Count, (double)rank5 / probes.Count, probes.Count, gallery.Count, identities);
    }
}
