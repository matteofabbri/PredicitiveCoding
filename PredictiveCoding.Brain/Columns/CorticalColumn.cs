namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// CorticalColumn  —  single sensorimotor cortical column (Phase 4).
//
// Integrates the Phase 0-3 building blocks into one learning + inference unit:
//
//   1. SpatialPooler  — maps raw sensor input to a stable, fixed-sparsity SDR.
//   2. TemporalMemory — learns feature sequences conditioned on location context.
//   3. ObjectMemory   — stores (location, feature) pairs per object across steps.
//   4. Hypothesis tracking — monotonically intersects per-step candidate sets to
//      converge on a unique object id.
//
// Learning flow
//   Call StartNewObject() for each object to learn, then Compute(…, learn:true)
//   for every (feature, location) observation on that object.  Or use the
//   LearnObject() convenience wrapper.
//
// Inference flow
//   Call Reset() to clear temporal context and hypothesis state, then
//   Compute(…, learn:false) for each sensation; inspect Converged on the output.
//
// Reference: Hawkins et al. (2019) "A Framework for Intelligence and Cortical
// Function Based on Grid Cells"; Leadholm et al. (2025) Sections 4-5.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record CorticalColumnConfig
{
    /// <summary>Raw sensory input size (SDR size from the upstream encoder).</summary>
    public required int InputSize { get; init; }

    /// <summary>Number of SpatialPooler output columns = TemporalMemory column count.</summary>
    public required int Columns { get; init; }

    public int CellsPerColumn { get; init; } = 8;

    /// <summary>Target fraction of active columns produced by the SpatialPooler.</summary>
    public float Sparsity { get; init; } = 0.02f;

    /// <summary>Minimum SDR overlap for a stored (location, feature) point to count as a match.</summary>
    public int MinOverlap { get; init; } = 3;

    public int Seed { get; init; } = 42;
}

/// <summary>Per-step output of the cortical column.</summary>
public sealed record CorticalColumnOutput(
    /// <summary>L4 active cells SDR (size = Columns × CellsPerColumn).</summary>
    Sdr ActiveCells,
    /// <summary>Object ids consistent with all observations so far.</summary>
    IReadOnlyList<int> ObjectHypotheses,
    /// <summary>True when exactly one hypothesis remains (object identified).</summary>
    bool Converged,
    /// <summary>Fraction of columns that burst this step (0 = full prediction, 1 = total surprise).</summary>
    float Anomaly);

public sealed class CorticalColumn
{
    private readonly CorticalColumnConfig _cfg;
    private readonly SpatialPooler _sp;
    private readonly TemporalMemory _tm;
    private readonly ObjectMemory _objectMemory;

    private HashSet<int> _hypotheses = [];
    private int _inferenceSteps;
    private int _nextObjectId;
    private int _learningObjectId = -1;

    /// <summary>Number of distinct objects stored in memory.</summary>
    public int ObjectCount => _objectMemory.ObjectCount;

    /// <summary>Current inference hypothesis set (updated by each <see cref="Compute"/> call).</summary>
    public IReadOnlyCollection<int> CurrentHypotheses => _hypotheses;

    public CorticalColumn(CorticalColumnConfig cfg)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        if (cfg.InputSize <= 0 || cfg.Columns <= 0)
            throw new ArgumentException("InputSize and Columns must be positive.", nameof(cfg));
        _cfg = cfg;

        int activeColumns = Math.Max(1, (int)Math.Round(cfg.Sparsity * cfg.Columns));
        _sp = new SpatialPooler(new SpatialPoolerConfig
        {
            InputSize = cfg.InputSize,
            Columns = cfg.Columns,
            ActiveColumns = activeColumns,
            Seed = cfg.Seed,
        });
        _tm = new TemporalMemory(new TemporalMemoryConfig
        {
            Columns = cfg.Columns,
            CellsPerColumn = cfg.CellsPerColumn,
            Seed = cfg.Seed + 1,
        });
        _objectMemory = new ObjectMemory(cfg.MinOverlap);
    }

    /// <summary>Begins a new learning episode for a fresh object.
    /// Must be called before any <see cref="Compute"/> with <c>learn=true</c>.
    /// Returns the auto-assigned object id.</summary>
    public int StartNewObject()
    {
        _learningObjectId = _nextObjectId++;
        return _learningObjectId;
    }

    /// <summary>Processes one sensorimotor step.
    /// <paramref name="rawFeature"/> must have Size = InputSize.
    /// <paramref name="location"/> is the current location SDR from a PoseEncoder or GridCellEncoder.
    /// When <paramref name="learn"/> is true the observation is stored under the active learning object.</summary>
    public CorticalColumnOutput Compute(Sdr rawFeature, Sdr location, bool learn)
    {
        ArgumentNullException.ThrowIfNull(rawFeature);
        ArgumentNullException.ThrowIfNull(location);
        if (rawFeature.Size != _cfg.InputSize)
            throw new ArgumentException(
                $"rawFeature.Size={rawFeature.Size} != InputSize={_cfg.InputSize}.", nameof(rawFeature));

        // Stage 1: SpatialPooler → stable feature columns SDR.
        var featureCols = _sp.Compute(rawFeature, learn);

        // Stage 2: TemporalMemory → context-rich cell representation + anomaly.
        var tmResult = _tm.Compute(featureCols, learn);

        // Stage 3: store (learn) or intersect hypotheses (infer).
        IReadOnlyList<int> hypotheses;
        if (learn)
        {
            if (_learningObjectId < 0)
                throw new InvalidOperationException(
                    "Call StartNewObject() before calling Compute with learn=true.");
            _objectMemory.AddPoint(_learningObjectId, location, featureCols);
            hypotheses = [_learningObjectId];
        }
        else
        {
            var candidates = _objectMemory.IntersectingObjectIds(location, featureCols);
            if (_inferenceSteps == 0)
                _hypotheses = new HashSet<int>(candidates);
            else
                _hypotheses.IntersectWith(candidates);
            _inferenceSteps++;
            hypotheses = [.. _hypotheses];
        }

        var activeCellSdr = new Sdr(_cfg.Columns * _cfg.CellsPerColumn, tmResult.ActiveCells);
        return new CorticalColumnOutput(
            activeCellSdr,
            hypotheses,
            hypotheses.Count == 1,
            tmResult.Anomaly);
    }

    /// <summary>Constrains the active hypothesis set to the intersection with
    /// <paramref name="candidates"/>. Called by <see cref="MultiColumnSystem"/> to inject
    /// cross-column consensus so subsequent steps start from the narrowed set.</summary>
    public void ConstrainHypotheses(IEnumerable<int> candidates)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        _hypotheses.IntersectWith(candidates);
    }

    /// <summary>Resets temporal context and the inference hypothesis set.
    /// Learned object memory is NOT cleared.</summary>
    public void Reset()
    {
        _tm.Reset();
        _hypotheses.Clear();
        _inferenceSteps = 0;
    }

    /// <summary>Convenience: starts a new object, resets temporal context, records all
    /// <paramref name="observations"/>, and returns the assigned object id.</summary>
    public int LearnObject(IReadOnlyList<(Sdr Feature, Sdr Location)> observations)
    {
        ArgumentNullException.ThrowIfNull(observations);
        int id = StartNewObject();
        _tm.Reset();
        foreach (var (feat, loc) in observations)
            Compute(feat, loc, learn: true);
        return id;
    }

    /// <summary>Runs a full inference sequence over <paramref name="observations"/>.
    /// Calls <see cref="Reset"/> before the first step and returns all per-step outputs.</summary>
    public IReadOnlyList<CorticalColumnOutput> Infer(
        IReadOnlyList<(Sdr Feature, Sdr Location)> observations)
    {
        ArgumentNullException.ThrowIfNull(observations);
        Reset();
        var results = new List<CorticalColumnOutput>(observations.Count);
        foreach (var (feat, loc) in observations)
            results.Add(Compute(feat, loc, learn: false));
        return results;
    }
}
