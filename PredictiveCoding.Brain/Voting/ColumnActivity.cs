namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// ColumnActivity  —  snapshot of one column's state after a compute step.
//
// Passed to the Consensus and LateralVoting components so they can operate on
// all columns' outputs without needing direct access to each column instance.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record ColumnActivity(
    /// <summary>Index of the column within the MultiColumnSystem.</summary>
    int ColumnIndex,

    /// <summary>L4 active-cells SDR (size = Columns × CellsPerColumn). Used as the
    /// lateral broadcast signal — other columns receive the union of their neighbours'
    /// ActiveCells SDRs as context for their next L2/3 computation.</summary>
    Sdr ActiveCells,

    /// <summary>Object ids consistent with all observations this column has accumulated
    /// so far. The Consensus component intersects these across all columns.</summary>
    IReadOnlyList<int> ObjectHypotheses,

    /// <summary>True when this column alone has converged to a single object id.</summary>
    bool Converged,

    /// <summary>Fraction of columns that burst this step (0 = full prediction).</summary>
    float Anomaly);
