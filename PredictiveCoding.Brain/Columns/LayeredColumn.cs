namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// LayeredColumn  —  canonical four-layer cortical microcircuit scaffold.
//
// Models the canonical neocortical column with four functionally distinct layers:
//
//   L4  — feedforward thalamic input; learns feature sequences via HTM TM.
//   L2/3 — object-level representation; receives L4 output + lateral connections
//           from sibling columns; learns object-level sequences.
//   L5  — motor/predictive output; simplified as a projection of L2/3.
//   L6a — location/context layer; encodes motor commands as categorical SDRs
//          and feeds them back as context for L4.
//
// L4 and L2/3 each carry an independent TemporalMemory. L5 currently mirrors
// L2/3 (full predictive L5 is scheduled for Phase 5). L6a uses a CategoryEncoder
// to represent discrete motor commands.
//
// Reference: Harris & Shepherd (2015) "Neocortical circuit: themes and variations."
// ─────────────────────────────────────────────────────────────────────────────

public sealed record LayeredColumnConfig
{
    /// <summary>Number of columns in L4 and L2/3 (= SpatialPooler output size).</summary>
    public required int FeatureColumns { get; init; }

    public int CellsPerColumn { get; init; } = 8;

    /// <summary>Number of discrete motor command categories for L6a encoding.</summary>
    public int MotorCategories { get; init; } = 16;

    /// <summary>Active bits per motor category in the L6a SDR.</summary>
    public int MotorBitsPerCategory { get; init; } = 4;

    /// <summary>Base seed for internal RNGs (L23 uses Seed+1).</summary>
    public int Seed { get; init; } = 42;
}

/// <summary>Per-step outputs of all four cortical layers.</summary>
public sealed record LayeredColumnOutput(Sdr L4, Sdr L23, Sdr L5, Sdr L6);

public sealed class LayeredColumn
{
    private readonly LayeredColumnConfig _cfg;
    private readonly TemporalMemory _l4Tm;
    private readonly TemporalMemory _l23Tm;
    private readonly CategoryEncoder _motorEncoder;

    private Sdr _l4Output;
    private Sdr _l23Output;

    /// <summary>Cell-level SDR output of L4 from the last <see cref="ComputeL4"/> call.</summary>
    public Sdr L4Output => _l4Output;

    /// <summary>Cell-level SDR output of L2/3 from the last <see cref="ComputeL23"/> call.</summary>
    public Sdr L23Output => _l23Output;

    public LayeredColumn(LayeredColumnConfig cfg)
    {
        ArgumentNullException.ThrowIfNull(cfg);
        if (cfg.FeatureColumns <= 0)
            throw new ArgumentException("FeatureColumns must be positive.", nameof(cfg));
        _cfg = cfg;

        var tmBase = new TemporalMemoryConfig
        {
            Columns = cfg.FeatureColumns,
            CellsPerColumn = cfg.CellsPerColumn,
            Seed = cfg.Seed,
        };
        _l4Tm = new TemporalMemory(tmBase);
        _l23Tm = new TemporalMemory(tmBase with { Seed = cfg.Seed + 1 });
        _motorEncoder = new CategoryEncoder(cfg.MotorCategories, cfg.MotorBitsPerCategory);

        var empty = Sdr.Empty(cfg.FeatureColumns * cfg.CellsPerColumn);
        _l4Output = empty;
        _l23Output = empty;
    }

    /// <summary>Runs the L4 TemporalMemory on thalamic feedforward input.
    /// <paramref name="thalamicColumns"/> must be a column-level SDR (Size = FeatureColumns).</summary>
    public Sdr ComputeL4(Sdr thalamicColumns, bool learn)
    {
        ArgumentNullException.ThrowIfNull(thalamicColumns);
        RequireColumnSdr(thalamicColumns, nameof(thalamicColumns));
        int cellSize = _cfg.FeatureColumns * _cfg.CellsPerColumn;
        _l4Output = new Sdr(cellSize, _l4Tm.Compute(thalamicColumns, learn).ActiveCells);
        return _l4Output;
    }

    /// <summary>Runs the L2/3 TemporalMemory.  Input is derived from the last L4 active cells
    /// projected back to column level, optionally unioned with lateral column activations from
    /// sibling columns in the same cortical area.</summary>
    public Sdr ComputeL23(Sdr? lateralColumns, bool learn)
    {
        var l4Cols = CellsToColumns(_l4Output, _cfg.CellsPerColumn, _cfg.FeatureColumns);
        var input = lateralColumns is null ? l4Cols : l4Cols.Union(ValidatedColumnSdr(lateralColumns));
        int cellSize = _cfg.FeatureColumns * _cfg.CellsPerColumn;
        _l23Output = new Sdr(cellSize, _l23Tm.Compute(input, learn).ActiveCells);
        return _l23Output;
    }

    /// <summary>L5 output: currently mirrors L2/3 active cells.
    /// A full predictive L5 (deep-layer motor output) is planned for Phase 5.</summary>
    public Sdr ComputeL5() => _l23Output;

    /// <summary>L6a motor-context SDR: encodes a discrete motor command as a categorical SDR
    /// that provides location/efference-copy context back to L4.</summary>
    public Sdr ComputeL6(int motorCategory) => _motorEncoder.Encode(motorCategory);

    /// <summary>Runs all four layers in the canonical order L4 → L2/3 → L5 → L6.</summary>
    public LayeredColumnOutput FullPass(
        Sdr thalamicColumns, Sdr? lateralColumns, int motorCategory, bool learn)
    {
        var l4 = ComputeL4(thalamicColumns, learn);
        var l23 = ComputeL23(lateralColumns, learn);
        var l5 = ComputeL5();
        var l6 = ComputeL6(motorCategory);
        return new LayeredColumnOutput(l4, l23, l5, l6);
    }

    /// <summary>Resets temporal context in both L4 and L2/3; clears cached outputs.</summary>
    public void Reset()
    {
        _l4Tm.Reset();
        _l23Tm.Reset();
        var empty = Sdr.Empty(_cfg.FeatureColumns * _cfg.CellsPerColumn);
        _l4Output = empty;
        _l23Output = empty;
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private void RequireColumnSdr(Sdr sdr, string paramName)
    {
        if (sdr.Size != _cfg.FeatureColumns)
            throw new ArgumentException(
                $"SDR size {sdr.Size} != FeatureColumns {_cfg.FeatureColumns}.", paramName);
    }

    private Sdr ValidatedColumnSdr(Sdr sdr)
    {
        RequireColumnSdr(sdr, nameof(sdr));
        return sdr;
    }

    /// <summary>Projects a cell-level SDR to column level: a column is active if any cell fires.</summary>
    private static Sdr CellsToColumns(Sdr cellSdr, int cellsPerColumn, int columns)
    {
        var activeCols = new HashSet<int>();
        foreach (int cell in cellSdr.ActiveBits)
            activeCols.Add(cell / cellsPerColumn);
        var arr = activeCols.ToArray();
        Array.Sort(arr);
        return new Sdr(columns, arr);
    }
}
