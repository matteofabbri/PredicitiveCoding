namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// Consensus  —  cross-column object-hypothesis consensus.
//
// Aggregates the ObjectHypotheses sets from all columns and returns the subset
// of object ids that satisfy the minimum agreement threshold. With full
// unanimity (threshold=1.0), only objects that every column considers possible
// survive. With majority threshold (threshold=0.5), faster convergence is
// possible even when individual columns are noisy.
//
// The Compute method is pure (stateless); the MultiColumnSystem owns the
// convergence loop and calls Compute once per round.
//
// Reference: Hawkins (2021) Chapter 8; Leadholm et al. (2025) Section 5.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class Consensus
{
    /// <summary>Minimum fraction of columns that must include an object id for it to
    /// survive as a candidate (0.5 = majority; 1.0 = unanimity).</summary>
    public float MinAgreementFraction { get; }

    /// <param name="minAgreementFraction">Agreement threshold in (0, 1]. Default 1.0 (unanimity).</param>
    public Consensus(float minAgreementFraction = 1.0f)
    {
        if (minAgreementFraction <= 0f || minAgreementFraction > 1f)
            throw new ArgumentOutOfRangeException(nameof(minAgreementFraction), "Must be in (0, 1].");
        MinAgreementFraction = minAgreementFraction;
    }

    /// <summary>Computes cross-column consensus for one voting round.
    /// Returns all candidate object ids whose agreement score ≥ <see cref="MinAgreementFraction"/>.</summary>
    public ConsensusResult Compute(IReadOnlyList<ColumnActivity> activities, int round = 0)
    {
        ArgumentNullException.ThrowIfNull(activities);
        int n = activities.Count;
        if (n == 0) return new ConsensusResult([], 0f, false, round);

        var votes = CountVotes(activities);
        var candidates = votes
            .Where(kv => (float)kv.Value / n >= MinAgreementFraction)
            .OrderBy(kv => kv.Key)
            .Select(kv => kv.Key)
            .ToList();

        float score = candidates.Count > 0
            ? candidates.Average(id => (float)votes[id] / n)
            : 0f;

        return new ConsensusResult(candidates, score, candidates.Count > 0, round);
    }

    /// <summary>Agreement score for a specific object id: fraction of columns that include it
    /// in their current hypothesis set.</summary>
    public float AgreementScore(int objectId, IReadOnlyList<ColumnActivity> activities)
    {
        ArgumentNullException.ThrowIfNull(activities);
        if (activities.Count == 0) return 0f;
        int votes = activities.Count(a => a.ObjectHypotheses.Contains(objectId));
        return (float)votes / activities.Count;
    }

    /// <summary>Returns the top <paramref name="topK"/> object ids ranked by how many columns
    /// include them, regardless of the agreement threshold. Useful when the strict threshold
    /// yields an empty candidate set.</summary>
    public IReadOnlyList<int> TopCandidatesByAgreement(
        IReadOnlyList<ColumnActivity> activities, int topK = 5)
    {
        ArgumentNullException.ThrowIfNull(activities);
        var votes = CountVotes(activities);
        return votes
            .OrderByDescending(kv => kv.Value)
            .Take(topK)
            .Select(kv => kv.Key)
            .ToList();
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static Dictionary<int, int> CountVotes(IReadOnlyList<ColumnActivity> activities)
    {
        var votes = new Dictionary<int, int>();
        foreach (var act in activities)
            foreach (int id in act.ObjectHypotheses)
                votes[id] = votes.GetValueOrDefault(id) + 1;
        return votes;
    }
}
