namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// MultiColumnSystem  —  N cortical columns with lateral voting.
//
// Orchestrates a population of CorticalColumn instances that each observe the
// same object from different vantage points or via different sensory modalities.
// After each sensorimotor step every column shares its active-cell SDR with its
// neighbours via LateralVoting, and the cross-column hypothesis consensus is
// computed by the Consensus module. Converging to a unique object id typically
// takes far fewer steps than a single isolated column.
//
// Learning flow
//   StartNewObject() once → LearnObject(perColumnObs) or manual calls to
//   ComputeStep(…, learn:true) per observation.
//
// Inference flow
//   Reset() → one or more VoteStep() calls → watch CurrentConsensus.IsReached.
//   Or call RunToConsensus() which loops internally.
//
// Reference: Hawkins (2021) Chapters 7-8; Leadholm et al. (2025) Section 5.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class MultiColumnSystem
{
    private readonly CorticalColumn[] _columns;
    private readonly LateralVoting _lateral;
    private readonly Consensus _consensus;
    private ConsensusResult? _currentConsensus;

    /// <summary>Number of columns in the system.</summary>
    public int ColumnCount => _columns.Length;

    /// <summary>Number of learned objects (from the first column as the reference).</summary>
    public int ObjectCount => _columns.Length > 0 ? _columns[0].ObjectCount : 0;

    /// <summary>Consensus result from the most recent <see cref="VoteStep"/> call; null before the first call.</summary>
    public ConsensusResult? CurrentConsensus => _currentConsensus;

    /// <param name="columnConfigs">One config per column. All columns must have the same
    /// InputSize and Columns so their active-cell SDRs are union-compatible.</param>
    /// <param name="topology">Lateral connection pattern. Default: AllToAll.</param>
    /// <param name="minAgreementFraction">Agreement threshold for the Consensus module. Default: 0.5.</param>
    public MultiColumnSystem(
        IReadOnlyList<CorticalColumnConfig> columnConfigs,
        VotingTopology topology = VotingTopology.AllToAll,
        float minAgreementFraction = 0.5f)
    {
        ArgumentNullException.ThrowIfNull(columnConfigs);
        if (columnConfigs.Count < 2)
            throw new ArgumentException("At least 2 column configs required.", nameof(columnConfigs));

        _columns = columnConfigs.Select(c => new CorticalColumn(c)).ToArray();
        _lateral = new LateralVoting(_columns.Length, topology);
        _consensus = new Consensus(minAgreementFraction);
    }

    /// <summary>Starts a new object learning episode on all columns simultaneously.
    /// Returns the shared object id (auto-incremented in lock-step across all columns).</summary>
    public int StartNewObject()
    {
        int id = _columns[0].StartNewObject();
        for (int i = 1; i < _columns.Length; i++)
        {
            int otherId = _columns[i].StartNewObject();
            if (otherId != id)
                throw new InvalidOperationException(
                    $"Column {i} assigned object id {otherId} but column 0 assigned {id}. " +
                    "Columns must always have StartNewObject called together to stay in sync.");
        }
        return id;
    }

    /// <summary>Convenience: starts a new object and feeds per-column observation lists to each column.
    /// <paramref name="perColumnObservations"/>[i] is the sequence of (feature, location) pairs for column i.</summary>
    public int LearnObject(
        IReadOnlyList<IReadOnlyList<(Sdr Feature, Sdr Location)>> perColumnObservations)
    {
        ArgumentNullException.ThrowIfNull(perColumnObservations);
        if (perColumnObservations.Count != _columns.Length)
            throw new ArgumentException(
                $"Expected {_columns.Length} observation lists, got {perColumnObservations.Count}.",
                nameof(perColumnObservations));

        int id = StartNewObject();
        for (int i = 0; i < _columns.Length; i++)
        {
            _columns[i].Reset();
            foreach (var (feat, loc) in perColumnObservations[i])
                _columns[i].Compute(feat, loc, learn: true);
        }
        return id;
    }

    /// <summary>Runs one step on all columns. Returns one <see cref="ColumnActivity"/> per column.</summary>
    public IReadOnlyList<ColumnActivity> ComputeStep(
        IReadOnlyList<(Sdr Feature, Sdr Location)> inputs, bool learn)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        if (inputs.Count != _columns.Length)
            throw new ArgumentException(
                $"Expected {_columns.Length} inputs, got {inputs.Count}.", nameof(inputs));

        var activities = new ColumnActivity[_columns.Length];
        for (int i = 0; i < _columns.Length; i++)
        {
            var (feat, loc) = inputs[i];
            var out_ = _columns[i].Compute(feat, loc, learn);
            activities[i] = new ColumnActivity(i, out_.ActiveCells, out_.ObjectHypotheses,
                out_.Converged, out_.Anomaly);
        }
        return activities;
    }

    /// <summary>Runs one voting round: computes a step on all columns, exchanges lateral signals,
    /// applies the cross-column consensus to narrow each column's hypotheses, and returns the result.</summary>
    public ConsensusResult VoteStep(IReadOnlyList<(Sdr Feature, Sdr Location)> inputs)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        var activities = ComputeStep(inputs, learn: false);

        // Lateral SDR broadcast (biologically: L2/3 horizontal connections).
        var activeCells = activities.Select(a => a.ActiveCells).ToList();
        var lateralSdrs = _lateral.Broadcast(activeCells);   // one per column

        // Hypothesis-level consensus.
        _currentConsensus = _consensus.Compute(activities);

        // Apply the consensus back to each column to narrow future inference.
        if (_currentConsensus.Candidates.Count > 0)
            foreach (var col in _columns)
                col.ConstrainHypotheses(_currentConsensus.Candidates);

        return _currentConsensus;
    }

    /// <summary>Resets temporal context and hypothesis state in all columns.</summary>
    public void Reset()
    {
        foreach (var col in _columns)
            col.Reset();
        _currentConsensus = null;
    }

    /// <summary>Runs <see cref="VoteStep"/> up to <paramref name="maxRounds"/> times until
    /// the consensus reaches a single candidate. Returns the final consensus.</summary>
    public ConsensusResult RunToConsensus(
        IReadOnlyList<(Sdr Feature, Sdr Location)> inputs, int maxRounds = 20)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        if (maxRounds < 1) throw new ArgumentOutOfRangeException(nameof(maxRounds));

        ConsensusResult result = new([], 0f, false, 0);
        for (int r = 0; r < maxRounds; r++)
        {
            result = VoteStep(inputs);
            if (result.Candidates.Count == 1) break;
        }
        return result;
    }
}
