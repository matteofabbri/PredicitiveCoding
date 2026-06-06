namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// KWinners  —  k-winners-take-all inhibition utilities.
//
// Selects the top-k scoring units from a population, enforcing sparse activity.
// LocalInhibition additionally prevents neighbouring winners from clustering
// by applying per-position neighbourhood competitions.
//
// Reference: Ahmad & Hawkins (2016) "How do neurons operate on sparse
// distributed representations?" Numenta research paper.
// ─────────────────────────────────────────────────────────────────────────────

public static class KWinners
{
    /// <summary>Indices of the top-<paramref name="k"/> entries in <paramref name="scores"/>;
    /// ties resolved by lower index first. Result is sorted ascending.</summary>
    public static int[] SelectTopK(IReadOnlyList<double> scores, int k)
    {
        ArgumentNullException.ThrowIfNull(scores);
        if (k <= 0 || scores.Count == 0) return [];

        int take = Math.Min(k, scores.Count);
        var indices = new int[scores.Count];
        for (int i = 0; i < indices.Length; i++) indices[i] = i;

        Array.Sort(indices, (a, b) =>
        {
            int cmp = scores[b].CompareTo(scores[a]);   // descending by score
            return cmp != 0 ? cmp : a.CompareTo(b);     // ascending index on ties
        });

        var result = new int[take];
        Array.Copy(indices, result, take);
        Array.Sort(result);
        return result;
    }

    /// <summary>Indices of the top <paramref name="sparsity"/> fraction of units
    /// (e.g. 0.02 → top 2%). Always returns at least one winner.</summary>
    public static int[] SelectTopKPercent(IReadOnlyList<double> scores, double sparsity)
    {
        ArgumentNullException.ThrowIfNull(scores);
        if (sparsity <= 0d || scores.Count == 0) return [];
        int k = Math.Max(1, (int)Math.Round(sparsity * scores.Count));
        return SelectTopK(scores, k);
    }

    /// <summary>Local inhibition: position <c>i</c> wins only if it ranks among the top
    /// <paramref name="kPerRadius"/> scores within the window <c>[i−radius, i+radius]</c>.
    /// Distributes winners spatially rather than letting them cluster in one region.</summary>
    public static int[] LocalInhibition(IReadOnlyList<double> scores, int radius, int kPerRadius)
    {
        ArgumentNullException.ThrowIfNull(scores);
        if (kPerRadius <= 0 || scores.Count == 0) return [];

        int n = scores.Count;
        var winners = new List<int>(n);

        for (int center = 0; center < n; center++)
        {
            int lo = Math.Max(0, center - radius);
            int hi = Math.Min(n - 1, center + radius);

            // Rank of this position within its neighbourhood (0 = best score).
            int rank = 0;
            for (int j = lo; j <= hi; j++)
            {
                if (scores[j] > scores[center]) rank++;
                else if (scores[j] == scores[center] && j < center) rank++;
            }

            if (rank < kPerRadius) winners.Add(center);
        }

        return winners.ToArray();
    }
}
