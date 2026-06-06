namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// CrossModalSystem  —  voting across sensory modalities into a shared object id.
//
// A Thousand-Brains claim (Hawkins 2021): vision, touch and other senses each model
// an object in their own frame, yet converge on a single shared identity by voting.
// MultiColumnSystem assumes homogeneous columns because its lateral pathway unions
// active-cell SDRs (which must be the same size). Cross-modal voting needs no such
// union: the consensus is computed over object-id hypotheses, which are
// modality-agnostic. So this system holds one CorticalColumn per modality — each
// free to use a different feature encoder and InputSize — and lets them vote purely
// at the hypothesis level. Either modality alone may stay ambiguous; together they
// disambiguate.
//
// Reference: Hawkins (2021) P1 Ch. 7; Leadholm et al. (2025) P9 (multi-sensor voting).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class CrossModalSystem
{
    private readonly CorticalColumn[] _columns;   // one per modality (heterogeneous InputSize allowed)
    private readonly Consensus _consensus;
    private ConsensusResult? _current;

    /// <summary>Number of sensory modalities.</summary>
    public int ModalityCount => _columns.Length;

    /// <summary>Number of learned objects (shared across all modalities).</summary>
    public int ObjectCount => _columns.Length > 0 ? _columns[0].ObjectCount : 0;

    /// <summary>Consensus from the most recent <see cref="VoteStep"/>; null before the first call.</summary>
    public ConsensusResult? CurrentConsensus => _current;

    /// <param name="modalityConfigs">One column config per modality; InputSize may differ between them.</param>
    /// <param name="minAgreementFraction">Agreement threshold for the consensus. Default 1.0
    /// (unanimous = strict intersection: all modalities must agree on the identity).</param>
    public CrossModalSystem(IReadOnlyList<CorticalColumnConfig> modalityConfigs, float minAgreementFraction = 1.0f)
    {
        ArgumentNullException.ThrowIfNull(modalityConfigs);
        if (modalityConfigs.Count < 2)
            throw new ArgumentException("At least 2 modalities are required.", nameof(modalityConfigs));

        _columns = modalityConfigs.Select(c => new CorticalColumn(c)).ToArray();
        _consensus = new Consensus(minAgreementFraction);
    }

    /// <summary>Teaches one object across all modalities. <paramref name="perModalityObservations"/>[m]
    /// is the (Feature, Location) sequence modality m sensed for this object. Object ids are assigned
    /// in lock-step so every modality shares the same id. Returns that id.</summary>
    public int LearnObject(
        IReadOnlyList<IReadOnlyList<(Sdr Feature, Sdr Location)>> perModalityObservations)
    {
        ArgumentNullException.ThrowIfNull(perModalityObservations);
        if (perModalityObservations.Count != _columns.Length)
            throw new ArgumentException(
                $"Expected {_columns.Length} modality observation lists, got {perModalityObservations.Count}.",
                nameof(perModalityObservations));

        int id = _columns[0].StartNewObject();
        for (int m = 1; m < _columns.Length; m++)
            if (_columns[m].StartNewObject() != id)
                throw new InvalidOperationException("Modalities fell out of id sync.");

        for (int m = 0; m < _columns.Length; m++)
        {
            _columns[m].Reset();
            foreach (var (feat, loc) in perModalityObservations[m])
                _columns[m].Compute(feat, loc, learn: true);
        }
        return id;
    }

    /// <summary>Runs one cross-modal voting round: each modality senses its own
    /// (Feature, Location), the consensus is taken over their object-id hypotheses,
    /// and the agreed candidates are fed back to constrain every modality.</summary>
    public ConsensusResult VoteStep(IReadOnlyList<(Sdr Feature, Sdr Location)> perModalityInputs)
    {
        ArgumentNullException.ThrowIfNull(perModalityInputs);
        if (perModalityInputs.Count != _columns.Length)
            throw new ArgumentException(
                $"Expected {_columns.Length} modality inputs, got {perModalityInputs.Count}.",
                nameof(perModalityInputs));

        var activities = new ColumnActivity[_columns.Length];
        for (int m = 0; m < _columns.Length; m++)
        {
            var (feat, loc) = perModalityInputs[m];
            var output = _columns[m].Compute(feat, loc, learn: false);
            activities[m] = new ColumnActivity(
                m, output.ActiveCells, output.ObjectHypotheses, output.Converged, output.Anomaly);
        }

        _current = _consensus.Compute(activities);
        if (_current.Candidates.Count > 0)
            foreach (var col in _columns)
                col.ConstrainHypotheses(_current.Candidates);
        return _current;
    }

    /// <summary>Resets temporal context and hypotheses in every modality.</summary>
    public void Reset()
    {
        foreach (var col in _columns) col.Reset();
        _current = null;
    }
}
