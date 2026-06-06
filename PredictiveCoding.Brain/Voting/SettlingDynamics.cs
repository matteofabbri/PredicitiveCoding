namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// SettlingDynamics  —  iterative convergence loop for multi-column voting.
//
// Drives a MultiColumnSystem through repeated VoteStep rounds until the
// hypothesis set either collapses to a unique object (IsReached = true) or the
// MaxRounds budget is exhausted. Records the full per-round history so callers
// can plot convergence curves or diagnose stuck hypotheses.
//
// Typical use: create once, call Settle for each inference query.
//
// Reference: Leadholm et al. (2025) Section 5 "settling dynamics".
// ─────────────────────────────────────────────────────────────────────────────

public sealed class SettlingDynamics
{
    /// <summary>Hard upper bound on voting rounds per Settle call.</summary>
    public int MaxRounds { get; }

    /// <summary>Agreement fraction forwarded to the system's Consensus module
    /// (informational; the threshold is set on the MultiColumnSystem at construction time).</summary>
    public float MinAgreementFraction { get; }

    /// <param name="maxRounds">Maximum voting rounds before giving up. Default 10.</param>
    /// <param name="minAgreementFraction">Informational threshold reference. Default 1.0.</param>
    public SettlingDynamics(int maxRounds = 10, float minAgreementFraction = 1.0f)
    {
        if (maxRounds < 1) throw new ArgumentOutOfRangeException(nameof(maxRounds));
        if (minAgreementFraction <= 0f || minAgreementFraction > 1f)
            throw new ArgumentOutOfRangeException(nameof(minAgreementFraction));
        MaxRounds = maxRounds;
        MinAgreementFraction = minAgreementFraction;
    }

    /// <summary>Calls <see cref="MultiColumnSystem.Reset"/>, then runs
    /// <see cref="MultiColumnSystem.VoteStep"/> up to <see cref="MaxRounds"/> times.
    /// Returns a full history and the convergence outcome.</summary>
    public SettlingResult Settle(
        MultiColumnSystem system,
        IReadOnlyList<(Sdr Feature, Sdr Location)> inputs)
    {
        ArgumentNullException.ThrowIfNull(system);
        ArgumentNullException.ThrowIfNull(inputs);

        system.Reset();
        var history = new List<ConsensusResult>(MaxRounds);
        int roundsToConverge = MaxRounds;
        bool converged = false;

        for (int r = 0; r < MaxRounds; r++)
        {
            var result = new ConsensusResult(
                system.VoteStep(inputs).Candidates,
                system.CurrentConsensus!.AgreementScore,
                system.CurrentConsensus!.IsReached,
                r);
            history.Add(result);

            if (result.Candidates.Count == 1)
            {
                roundsToConverge = r + 1;
                converged = true;
                break;
            }
        }

        var final = history.Count > 0 ? history[^1] : new ConsensusResult([], 0f, false, 0);
        return new SettlingResult(final, roundsToConverge, converged, history);
    }

    /// <summary>Same as <see cref="Settle"/> but does NOT call Reset first — useful when
    /// the system has already been primed with some learning steps and you want to continue
    /// from the current temporal context.</summary>
    public SettlingResult SettleFromCurrent(
        MultiColumnSystem system,
        IReadOnlyList<(Sdr Feature, Sdr Location)> inputs)
    {
        ArgumentNullException.ThrowIfNull(system);
        ArgumentNullException.ThrowIfNull(inputs);

        var history = new List<ConsensusResult>(MaxRounds);
        int roundsToConverge = MaxRounds;
        bool converged = false;

        for (int r = 0; r < MaxRounds; r++)
        {
            var step = system.VoteStep(inputs);
            var result = step with { Round = r };
            history.Add(result);

            if (result.Candidates.Count == 1)
            {
                roundsToConverge = r + 1;
                converged = true;
                break;
            }
        }

        var final = history.Count > 0 ? history[^1] : new ConsensusResult([], 0f, false, 0);
        return new SettlingResult(final, roundsToConverge, converged, history);
    }
}
