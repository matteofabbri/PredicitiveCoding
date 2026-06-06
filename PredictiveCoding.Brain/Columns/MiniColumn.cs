namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// MiniColumn  —  cortical mini-column abstraction.
//
// A mini-column groups a fixed number of cells sharing the same feedforward
// receptive field. Cells within the column represent *context* (which sequence
// preceded the input), so a column encodes WHAT while the active cell encodes
// WHEN/HOW. Activation modes:
//   • Predicted  — only contextually consistent cells fire (no surprise).
//   • Bursting   — all cells fire (column active but unpredicted; surprise).
//   • Inactive   — column not driven by current input.
//
// Reference: Harris & Shepherd (2015); Hawkins et al. (2017) Framework whitepaper.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class MiniColumn
{
    private int[] _activeCells;
    private bool _isBursting;

    /// <summary>Population index of this column.</summary>
    public int Index { get; }

    /// <summary>Number of cells per column.</summary>
    public int CellCount { get; }

    /// <summary>Local cell indices (0-based within this column) that are currently active.</summary>
    public IReadOnlyList<int> ActiveCells => _activeCells;

    /// <summary>True when the column burst (no prior prediction; surprise mode).</summary>
    public bool IsBursting => _isBursting;

    /// <summary>True when at least one cell is active.</summary>
    public bool IsActive => _activeCells.Length > 0;

    public MiniColumn(int index, int cellsPerColumn)
    {
        if (index < 0) throw new ArgumentOutOfRangeException(nameof(index));
        if (cellsPerColumn <= 0) throw new ArgumentOutOfRangeException(nameof(cellsPerColumn));
        Index = index;
        CellCount = cellsPerColumn;
        _activeCells = [];
    }

    /// <summary>Activates specific cells (non-bursting, prediction-confirmed mode).</summary>
    public void SetActive(IReadOnlyList<int> localCellIndices)
    {
        ArgumentNullException.ThrowIfNull(localCellIndices);
        foreach (int c in localCellIndices)
            if ((uint)c >= (uint)CellCount)
                throw new ArgumentOutOfRangeException(nameof(localCellIndices),
                    $"Cell {c} out of range [0, {CellCount}).");

        var copy = localCellIndices.ToArray();
        Array.Sort(copy);
        _activeCells = copy;
        _isBursting = false;
    }

    /// <summary>Activates all cells — signals surprise / no prior prediction.</summary>
    public void SetBursting()
    {
        var all = new int[CellCount];
        for (int i = 0; i < CellCount; i++) all[i] = i;
        _activeCells = all;
        _isBursting = true;
    }

    /// <summary>Deactivates the column and clears burst state.</summary>
    public void Reset()
    {
        _activeCells = [];
        _isBursting = false;
    }

    /// <summary>Maps a local cell index (0-based within this column) to the global cell index
    /// across the whole column population (= Index × CellCount + localCell).</summary>
    public int GlobalCellIndex(int localCell)
    {
        if ((uint)localCell >= (uint)CellCount)
            throw new ArgumentOutOfRangeException(nameof(localCell));
        return Index * CellCount + localCell;
    }

    /// <summary>True if the given global cell index belongs to this column.</summary>
    public bool Contains(int globalCellIndex)
    {
        int lo = Index * CellCount;
        return globalCellIndex >= lo && globalCellIndex < lo + CellCount;
    }

    public override string ToString()
        => $"MiniColumn[{Index}] {_activeCells.Length}/{CellCount}{(_isBursting ? " burst" : "")}";
}
