namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// TemporalMemory  —  HTM high-order sequence memory.
//
// Each column has several cells; the column that is active represents the current
// input, while *which* cell is active represents the temporal context (the
// sequence so far). Cells grow distal dendritic *segments* with synapses to other
// cells that were active on the previous step. A segment with enough connected
// active synapses puts its cell into a *predictive* state — a prediction of the
// next input. When an active column was predicted, only the predicted cells fire;
// when it was not, the whole column *bursts* (signalling surprise) and a winner
// cell is chosen to learn the new transition.
//
// This is the mechanism of Hawkins & Ahmad (2016), "Why Neurons Have Thousands of
// Synapses, a Theory of Sequence Memory in Neocortex" (Frontiers in Neural Circuits).
// ─────────────────────────────────────────────────────────────────────────────

public sealed record TemporalMemoryConfig
{
    public required int Columns { get; init; }
    public int CellsPerColumn { get; init; } = 8;

    /// <summary>Connected active synapses needed for a segment to be active (→ prediction).</summary>
    public int ActivationThreshold { get; init; } = 13;

    /// <summary>Active (potential) synapses needed for a segment to "match" (learning target while bursting).</summary>
    public int MinThreshold { get; init; } = 10;

    public float ConnectedPermanence { get; init; } = 0.5f;
    public float InitialPermanence { get; init; } = 0.21f;
    public float PermanenceIncrement { get; init; } = 0.10f;
    public float PermanenceDecrement { get; init; } = 0.10f;

    /// <summary>Penalty applied to segments that predicted a column that did not activate (0 = off).</summary>
    public float PredictedSegmentDecrement { get; init; } = 0.0f;

    /// <summary>Maximum new synapses grown on a learning segment per step.</summary>
    public int MaxNewSynapseCount { get; init; } = 20;

    public int Seed { get; init; } = 42;

    public void Validate()
    {
        if (Columns <= 0 || CellsPerColumn <= 0)
            throw new ArgumentException("Columns and CellsPerColumn must be positive.");
        if (ActivationThreshold <= 0 || MinThreshold <= 0)
            throw new ArgumentException("Thresholds must be positive.");
    }
}

/// <summary>Outcome of one <see cref="TemporalMemory.Compute"/> step.</summary>
public sealed record TemporalMemoryResult(
    IReadOnlyList<int> ActiveCells,
    IReadOnlyList<int> WinnerCells,
    IReadOnlyList<int> PredictiveCells,
    float Anomaly);

public sealed class TemporalMemory
{
    private sealed class Synapse(int presynapticCell, float permanence)
    {
        public int PresynapticCell { get; } = presynapticCell;
        public float Permanence { get; set; } = permanence;
    }

    private sealed class Segment(int cell)
    {
        public int Cell { get; } = cell;
        public List<Synapse> Synapses { get; } = [];
    }

    private readonly TemporalMemoryConfig _cfg;
    private readonly int _numCells;
    private readonly List<Segment>?[] _cellSegments;
    private readonly Random _rng;

    // State carried between steps.
    private HashSet<int> _prevActiveCells = [];
    private HashSet<int> _prevWinnerCells = [];
    private List<Segment> _activeSegments = [];
    private List<Segment> _matchingSegments = [];
    private Dictionary<Segment, int> _numActivePotential = [];

    public int Columns => _cfg.Columns;
    public int CellsPerColumn => _cfg.CellsPerColumn;

    public TemporalMemory(TemporalMemoryConfig cfg)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        cfg.Validate();
        _cfg = cfg;
        _numCells = cfg.Columns * cfg.CellsPerColumn;
        _cellSegments = new List<Segment>?[_numCells];
        _rng = new Random(cfg.Seed);
    }

    /// <summary>Clears all temporal context (call between independent sequences).</summary>
    public void Reset()
    {
        _prevActiveCells = [];
        _prevWinnerCells = [];
        _activeSegments = [];
        _matchingSegments = [];
        _numActivePotential = [];
    }

    /// <summary>Presents one step of active columns; returns the new cell activity and anomaly.</summary>
    public TemporalMemoryResult Compute(Sdr activeColumns, bool learn)
    {
        ArgumentNullException.ThrowIfNull(activeColumns);
        if (activeColumns.Size != _cfg.Columns)
            throw new ArgumentException($"activeColumns size {activeColumns.Size} != Columns {_cfg.Columns}.", nameof(activeColumns));

        var activeColumnSet = new HashSet<int>(activeColumns.ActiveBits);
        Dictionary<int, List<Segment>> activeByCol = GroupByColumn(_activeSegments);
        Dictionary<int, List<Segment>> matchingByCol = GroupByColumn(_matchingSegments);

        var activeCells = new HashSet<int>();
        var winnerCells = new HashSet<int>();
        int burstingColumns = 0;

        foreach (int col in activeColumns.ActiveBits)
        {
            if (activeByCol.TryGetValue(col, out var colActive) && colActive.Count > 0)
            {
                // Predicted column: activate the predicted cells only.
                foreach (var seg in colActive)
                {
                    activeCells.Add(seg.Cell);
                    winnerCells.Add(seg.Cell);
                }
                if (learn)
                    foreach (var seg in colActive)
                        LearnOnSegment(seg);
            }
            else
            {
                // Unpredicted column: burst, then pick a winner / learning segment.
                burstingColumns++;
                foreach (int cell in ColumnCells(col)) activeCells.Add(cell);

                Segment? learningSegment = null;
                int winnerCell;
                if (matchingByCol.TryGetValue(col, out var colMatching) && colMatching.Count > 0)
                {
                    learningSegment = colMatching
                        .OrderByDescending(s => _numActivePotential.GetValueOrDefault(s, 0))
                        .ThenBy(s => s.Cell)
                        .First();
                    winnerCell = learningSegment.Cell;
                }
                else
                {
                    winnerCell = LeastUsedCell(col);
                    if (learn && _prevWinnerCells.Count > 0)
                        learningSegment = CreateSegment(winnerCell);
                }

                winnerCells.Add(winnerCell);
                if (learn && learningSegment is not null)
                    LearnOnSegment(learningSegment);
            }
        }

        // Punish segments that predicted a column which did not activate.
        if (learn && _cfg.PredictedSegmentDecrement > 0f)
            foreach (var seg in _matchingSegments)
                if (!activeColumnSet.Contains(ColumnOf(seg.Cell)))
                    AdaptSegment(seg, _prevActiveCells, -_cfg.PredictedSegmentDecrement, 0f);

        // Compute predictions for the next step from the new active cells.
        var predictive = ActivateDendrites(activeCells, out var newActive, out var newMatching, out var newPotential);

        float anomaly = activeColumns.Cardinality == 0 ? 0f : (float)burstingColumns / activeColumns.Cardinality;

        _prevActiveCells = activeCells;
        _prevWinnerCells = winnerCells;
        _activeSegments = newActive;
        _matchingSegments = newMatching;
        _numActivePotential = newPotential;

        return new TemporalMemoryResult(Sorted(activeCells), Sorted(winnerCells), predictive, anomaly);
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    private void LearnOnSegment(Segment seg)
    {
        AdaptSegment(seg, _prevActiveCells, _cfg.PermanenceIncrement, _cfg.PermanenceDecrement);
        int growth = _cfg.MaxNewSynapseCount - _numActivePotential.GetValueOrDefault(seg, 0);
        if (growth > 0) GrowSynapses(seg, _prevWinnerCells, growth);
    }

    private static void AdaptSegment(Segment seg, HashSet<int> presynapticActive, float increment, float decrement)
    {
        foreach (var syn in seg.Synapses)
        {
            syn.Permanence = presynapticActive.Contains(syn.PresynapticCell)
                ? Math.Clamp(syn.Permanence + increment, 0f, 1f)
                : Math.Clamp(syn.Permanence - decrement, 0f, 1f);
        }
    }

    private void GrowSynapses(Segment seg, HashSet<int> candidateCells, int maxNew)
    {
        var existing = new HashSet<int>(seg.Synapses.Select(s => s.PresynapticCell));
        var candidates = new List<int>();
        foreach (int cell in candidateCells)
            if (cell != seg.Cell && !existing.Contains(cell)) candidates.Add(cell);

        // Fisher–Yates partial shuffle to pick up to maxNew distinct candidates.
        int n = Math.Min(maxNew, candidates.Count);
        for (int i = 0; i < n; i++)
        {
            int j = _rng.Next(i, candidates.Count);
            (candidates[i], candidates[j]) = (candidates[j], candidates[i]);
            seg.Synapses.Add(new Synapse(candidates[i], _cfg.InitialPermanence));
        }
    }

    private Segment CreateSegment(int cell)
    {
        var seg = new Segment(cell);
        (_cellSegments[cell] ??= []).Add(seg);
        return seg;
    }

    // ── Dendrite activation (prediction) ─────────────────────────────────────

    private List<int> ActivateDendrites(
        HashSet<int> activeCells,
        out List<Segment> active, out List<Segment> matching, out Dictionary<Segment, int> numActivePotential)
    {
        active = [];
        matching = [];
        numActivePotential = [];
        var predictive = new SortedSet<int>();

        foreach (var segList in _cellSegments)
        {
            if (segList is null) continue;
            foreach (var seg in segList)
            {
                int connectedActive = 0, potentialActive = 0;
                foreach (var syn in seg.Synapses)
                {
                    if (!activeCells.Contains(syn.PresynapticCell)) continue;
                    potentialActive++;
                    if (syn.Permanence >= _cfg.ConnectedPermanence) connectedActive++;
                }
                if (connectedActive >= _cfg.ActivationThreshold)
                {
                    active.Add(seg);
                    predictive.Add(seg.Cell);
                }
                if (potentialActive >= _cfg.MinThreshold) matching.Add(seg);
                if (potentialActive > 0) numActivePotential[seg] = potentialActive;
            }
        }
        return [.. predictive];
    }

    // ── Small helpers ────────────────────────────────────────────────────────

    private int ColumnOf(int cell) => cell / _cfg.CellsPerColumn;

    private IEnumerable<int> ColumnCells(int column)
    {
        int start = column * _cfg.CellsPerColumn;
        for (int i = 0; i < _cfg.CellsPerColumn; i++) yield return start + i;
    }

    private int LeastUsedCell(int column)
    {
        int start = column * _cfg.CellsPerColumn;
        int best = start, bestCount = int.MaxValue;
        for (int i = 0; i < _cfg.CellsPerColumn; i++)
        {
            int cell = start + i;
            int count = _cellSegments[cell]?.Count ?? 0;
            if (count < bestCount) { bestCount = count; best = cell; }
        }
        return best;
    }

    private Dictionary<int, List<Segment>> GroupByColumn(List<Segment> segments)
    {
        var map = new Dictionary<int, List<Segment>>();
        foreach (var seg in segments)
        {
            int col = ColumnOf(seg.Cell);
            if (!map.TryGetValue(col, out var list)) map[col] = list = [];
            list.Add(seg);
        }
        return map;
    }

    private static List<int> Sorted(HashSet<int> set)
    {
        var list = new List<int>(set);
        list.Sort();
        return list;
    }
}
