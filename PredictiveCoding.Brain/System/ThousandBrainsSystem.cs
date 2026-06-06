namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// ThousandBrainsSystem  —  top-level orchestrator for a Thousand-Brains network.
//
// Assembles a HeterarchicalGraph of MultiColumnSystem areas, wires long-range
// inter-area connections according to the configured topology, and exposes a
// clean teach / process / recognize API that hides the multi-area plumbing.
//
// Architecture
//   SensoryAreas  ×  ColumnsPerArea  CorticalColumns
//   Each area ← MultiColumnSystem with intra-area lateral voting
//   Areas ← HeterarchicalGraph with inter-area long-range edges
//   Belief ← iterative HeterarchicalMessagePassing on every observation
//
// Hypothesis accumulation
//   Reset() clears all column-level hypothesis sets.
//   ProcessObservation() calls PassMessages(resetColumnSystems: false) so that
//   each new sensation further restricts the hypothesis set through intersection
//   — the same monotone-narrowing logic used within a single CorticalColumn.
//
// Reference: Leadholm et al. (2025) P9; Hawkins (2021) P1 Chapter 7.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Top-level orchestrator that assembles cortical areas into a
/// Thousand-Brains sensorimotor system.</summary>
public sealed class ThousandBrainsSystem
{
    private readonly HeterarchicalGraph _graph;
    private readonly HeterarchicalMessagePassing _mp;
    private readonly ThousandBrainsConfig _config;
    private IReadOnlyList<int> _currentHypotheses = [];
    private int _learnedObjectCount;

    // ── Properties ───────────────────────────────────────────────────────────

    /// <summary>The underlying cortical-area graph.  Exposed for advanced use
    /// (e.g. inspecting individual areas, adding custom connections).</summary>
    public HeterarchicalGraph Graph => _graph;

    /// <summary>Number of objects successfully taught via <see cref="TeachObject"/>.</summary>
    public int LearnedObjectCount => _learnedObjectCount;

    /// <summary>Hypothesis set computed after the most recent
    /// <see cref="ProcessObservation"/> call.  Empty before the first call
    /// and after <see cref="Reset"/>.</summary>
    public IReadOnlyCollection<int> CurrentHypotheses => _currentHypotheses;

    // ── Construction ─────────────────────────────────────────────────────────

    /// <param name="config">Full system configuration. Validated on construction.</param>
    public ThousandBrainsSystem(ThousandBrainsConfig config)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (config.SensoryAreas < 1)
            throw new ArgumentException("At least one sensory area is required.", nameof(config));
        if (config.ColumnsPerArea < 2)
            throw new ArgumentException(
                "ColumnsPerArea must be ≥ 2 (MultiColumnSystem requires at least 2 columns).",
                nameof(config));

        _config = config;
        _graph = new HeterarchicalGraph();
        _mp = new HeterarchicalMessagePassing(
            config.MaxInferenceRounds,
            config.ConvergenceThreshold);

        // Create one MultiColumnSystem per sensory area.
        for (int a = 0; a < config.SensoryAreas; a++)
        {
            var columnConfigs = Enumerable.Range(0, config.ColumnsPerArea)
                .Select(c => config.ColumnConfig with
                {
                    Seed = config.Seed + a * 100 + c,
                })
                .ToList();

            var sys = new MultiColumnSystem(
                columnConfigs,
                config.IntraAreaTopology,
                config.MinAgreementFraction);

            _graph.AddArea(a, sys, $"Area{a}");
        }

        // Wire inter-area long-range connections.
        if (config.SensoryAreas > 1)
            WireInterAreaConnections(config.InterAreaTopology, config.SensoryAreas);
    }

    /// <summary>Convenience factory — equivalent to <c>new ThousandBrainsSystem(config)</c>.</summary>
    public static ThousandBrainsSystem Create(ThousandBrainsConfig config) => new(config);

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>Teaches the system about one object by distributing
    /// <paramref name="observations"/> across all areas and their columns.
    /// Returns the auto-assigned object id (0-based, increments with each call).</summary>
    /// <param name="observations">Sequence of (Feature, Location) pairs collected
    /// while sensing the object.  At least 1 pair required.</param>
    public int TeachObject(IReadOnlyList<(Sdr Feature, Sdr Location)> observations)
    {
        ArgumentNullException.ThrowIfNull(observations);
        if (observations.Count == 0)
            throw new ArgumentException(
                "At least one observation is required to teach an object.", nameof(observations));

        int assignedId = -1;
        foreach (var area in _graph.Areas)
        {
            var perColumn = SplitForColumns(observations, _config.ColumnsPerArea);
            int id = area.ColumnSystem.LearnObject(perColumn);
            if (assignedId < 0) assignedId = id;
            // All areas must assign the same id (they are always incremented together).
        }

        _learnedObjectCount++;
        return assignedId;
    }

    // ── Inference ────────────────────────────────────────────────────────────

    /// <summary>Processes one (Feature, Location) observation.  Broadcasts the
    /// observation to all areas and runs heterarchical message passing.
    /// Updates <see cref="CurrentHypotheses"/> via cross-area intersection.
    /// Returns the full per-area output dictionary.</summary>
    /// <remarks>Does NOT reset column systems — call <see cref="Reset"/> at the start
    /// of each new recognition episode to clear accumulated temporal context.</remarks>
    public IReadOnlyDictionary<int, AreaOutput> ProcessObservation(Sdr feature, Sdr location)
    {
        ArgumentNullException.ThrowIfNull(feature);
        ArgumentNullException.ThrowIfNull(location);

        // Every column in every area receives the same (feature, location) pair.
        var areaInputs = new Dictionary<int, IReadOnlyList<(Sdr, Sdr)>>(
            _config.SensoryAreas);

        foreach (var area in _graph.Areas)
        {
            var inputs = new List<(Sdr, Sdr)>(_config.ColumnsPerArea);
            for (int c = 0; c < _config.ColumnsPerArea; c++)
                inputs.Add((feature, location));
            areaInputs[area.AreaId] = inputs;
        }

        // Accumulating mode: hypothesis sets are NOT reset between observations.
        var outputs = _mp.PassMessages(_graph, areaInputs, resetColumnSystems: false);

        // Global hypothesis = intersection of all areas' consensus candidates.
        _currentHypotheses = IntersectAllCandidates(outputs);
        return outputs;
    }

    /// <summary>Processes every observation in <paramref name="observations"/> in order,
    /// returning the recognized object id as soon as a unique hypothesis is reached.
    /// Returns <c>null</c> if no unique hypothesis is found after all observations.</summary>
    public int? Recognize(IReadOnlyList<(Sdr Feature, Sdr Location)> observations)
    {
        ArgumentNullException.ThrowIfNull(observations);
        Reset();
        foreach (var (feat, loc) in observations)
        {
            ProcessObservation(feat, loc);
            if (_currentHypotheses.Count == 1)
                return _currentHypotheses[0];
        }
        return _currentHypotheses.Count == 1 ? _currentHypotheses[0] : null;
    }

    // ── State management ─────────────────────────────────────────────────────

    /// <summary>Clears temporal context and hypothesis sets in every column of every area.
    /// Does NOT erase learned object models.  Call at the start of each inference episode.</summary>
    public void Reset()
    {
        foreach (var area in _graph.Areas)
            area.ColumnSystem.Reset();
        _currentHypotheses = [];
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    private void WireInterAreaConnections(VotingTopology topology, int areaCount)
    {
        switch (topology)
        {
            case VotingTopology.AllToAll:
                for (int a = 0; a < areaCount; a++)
                    for (int b = 0; b < areaCount; b++)
                        if (a != b)
                            _graph.Connect(a, b, ConnectionType.Lateral, 1.0f);
                break;

            case VotingTopology.Ring:
                for (int a = 0; a < areaCount; a++)
                    _graph.Connect(a, (a + 1) % areaCount, ConnectionType.Lateral, 1.0f);
                break;

            case VotingTopology.Local:
                for (int a = 0; a < areaCount; a++)
                {
                    if (a > 0)
                        _graph.Connect(a, a - 1, ConnectionType.Lateral, 1.0f);
                    if (a < areaCount - 1)
                        _graph.Connect(a, a + 1, ConnectionType.Lateral, 1.0f);
                }
                break;
        }
    }

    private static IReadOnlyList<int> IntersectAllCandidates(
        IReadOnlyDictionary<int, AreaOutput> outputs)
    {
        var nonEmpty = outputs.Values
            .Where(o => o.Consensus.Candidates.Count > 0)
            .Select(o => (IEnumerable<int>)o.Consensus.Candidates)
            .ToList();

        if (nonEmpty.Count == 0) return [];

        var intersection = new HashSet<int>(nonEmpty[0]);
        for (int i = 1; i < nonEmpty.Count; i++)
            intersection.IntersectWith(nonEmpty[i]);

        var result = intersection.ToArray();
        Array.Sort(result);
        return result;
    }

    /// <summary>Distributes <paramref name="observations"/> round-robin across
    /// <paramref name="columnCount"/> columns.  If there are fewer observations than
    /// columns, the extras are filled by cycling from the start.</summary>
    private static IReadOnlyList<IReadOnlyList<(Sdr Feature, Sdr Location)>> SplitForColumns(
        IReadOnlyList<(Sdr Feature, Sdr Location)> observations, int columnCount)
    {
        var result = new List<List<(Sdr, Sdr)>>(columnCount);
        for (int c = 0; c < columnCount; c++) result.Add([]);

        for (int i = 0; i < observations.Count; i++)
            result[i % columnCount].Add(observations[i]);

        // Guarantee every column gets at least one observation.
        for (int c = 0; c < columnCount; c++)
            if (result[c].Count == 0)
                result[c].Add(observations[c % observations.Count]);

        return result;
    }
}
