namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// LateralVoting  —  lateral SDR exchange between cortical columns.
//
// Implements the long-range horizontal connections that are the defining
// mechanism of the Thousand Brains Theory (P1, P7). Each column's active-cell
// SDR is broadcast to its neighbours; each column receives the union of its
// neighbours' SDRs as a lateral context signal that biases its next L2/3 step.
//
// The three topologies give different speed/coverage trade-offs:
//   AllToAll — fastest convergence; O(N²) connections.
//   Ring     — minimal connections; models local cortical inhibition.
//   Local    — intermediate; each column hears a spatial neighbourhood.
//
// Reference: Hawkins (2021) Chapter 8; Hawkins et al. (2019) Section 3.4.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class LateralVoting
{
    private readonly int _n;
    private readonly VotingTopology _topology;
    private readonly int _localRadius;

    /// <summary>Number of columns in the system.</summary>
    public int ColumnCount => _n;

    /// <summary>Connection topology used to determine each column's neighbours.</summary>
    public VotingTopology Topology => _topology;

    /// <summary>Neighbourhood radius used when Topology is <see cref="VotingTopology.Local"/>.</summary>
    public int LocalRadius => _localRadius;

    /// <param name="columnCount">Total number of columns.</param>
    /// <param name="topology">Lateral connection pattern. Default: AllToAll.</param>
    /// <param name="localRadius">Radius for Local topology (ignored otherwise). Default: 2.</param>
    public LateralVoting(
        int columnCount,
        VotingTopology topology = VotingTopology.AllToAll,
        int localRadius = 2)
    {
        if (columnCount < 2)
            throw new ArgumentOutOfRangeException(nameof(columnCount), "At least 2 columns required.");
        if (localRadius < 1)
            throw new ArgumentOutOfRangeException(nameof(localRadius));
        _n = columnCount;
        _topology = topology;
        _localRadius = localRadius;
    }

    /// <summary>Returns the indices of columns that send lateral input to <paramref name="columnIndex"/>.</summary>
    public IReadOnlyList<int> Neighbors(int columnIndex)
    {
        if ((uint)columnIndex >= (uint)_n)
            throw new ArgumentOutOfRangeException(nameof(columnIndex));

        return _topology switch
        {
            VotingTopology.AllToAll => AllExcept(columnIndex),
            VotingTopology.Ring     => RingNeighbors(columnIndex),
            VotingTopology.Local    => LocalNeighbors(columnIndex),
            _                       => AllExcept(columnIndex),
        };
    }

    /// <summary>The lateral SDR that <paramref name="columnIndex"/> receives: the bitwise union
    /// of all its neighbours' active-cell SDRs.</summary>
    public Sdr ComputeLateralSdr(int columnIndex, IReadOnlyList<Sdr> allActiveCells)
    {
        ArgumentNullException.ThrowIfNull(allActiveCells);
        if (allActiveCells.Count != _n)
            throw new ArgumentException($"Expected {_n} SDRs, got {allActiveCells.Count}.", nameof(allActiveCells));

        var neighbours = Neighbors(columnIndex);
        if (neighbours.Count == 0) return Sdr.Empty(allActiveCells[columnIndex].Size);

        var sdr = allActiveCells[neighbours[0]];
        for (int i = 1; i < neighbours.Count; i++)
            sdr = sdr.Union(allActiveCells[neighbours[i]]);
        return sdr;
    }

    /// <summary>Computes the lateral SDR for every column simultaneously.
    /// <c>result[i]</c> is the union of all neighbour active-cell SDRs for column i.</summary>
    public IReadOnlyList<Sdr> Broadcast(IReadOnlyList<Sdr> allActiveCells)
    {
        ArgumentNullException.ThrowIfNull(allActiveCells);
        if (allActiveCells.Count != _n)
            throw new ArgumentException($"Expected {_n} SDRs, got {allActiveCells.Count}.", nameof(allActiveCells));

        var result = new Sdr[_n];
        for (int i = 0; i < _n; i++)
            result[i] = ComputeLateralSdr(i, allActiveCells);
        return result;
    }

    // ── Neighbourhood helpers ────────────────────────────────────────────────

    private IReadOnlyList<int> AllExcept(int col)
    {
        var list = new int[_n - 1];
        int idx = 0;
        for (int i = 0; i < _n; i++)
            if (i != col) list[idx++] = i;
        return list;
    }

    private IReadOnlyList<int> RingNeighbors(int col)
        => [(_n + col - 1) % _n, (col + 1) % _n];

    private IReadOnlyList<int> LocalNeighbors(int col)
    {
        var list = new List<int>(_localRadius * 2);
        for (int d = 1; d <= _localRadius; d++)
        {
            int left = ((_n + col - d) % _n);
            int right = (col + d) % _n;
            if (left != col) list.Add(left);
            if (right != col && right != left) list.Add(right);
        }
        list.Sort();
        return list;
    }
}
