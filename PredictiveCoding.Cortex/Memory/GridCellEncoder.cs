using PredictiveCoding.Core;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// GridCellEncoder  —  entorhinal grid-cell code for a 2-D position
//
// Encodes a continuous position (x, y) as a population vector of grid-cell
// activations.  Each grid module has a characteristic spatial scale and tiles
// space with a hexagonal pattern formed by three plane waves at 60°; cells within
// a module differ only in spatial phase.  Multiple modules at geometrically
// spaced scales give a high-capacity, periodic code from which position can be
// read out — the substrate for spatial predictive coding / path integration.
//
// Reference: Hafting et al. (2005); Fiete, Burak & Brookings (2008).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class GridCellEncoder
{
    private readonly int     _modules;
    private readonly int     _cellsPerModule;
    private readonly float[] _moduleFreq;     // spatial frequency per module
    private readonly float[] _phaseX;         // phase offset per cell
    private readonly float[] _phaseY;

    // Three plane-wave directions 60° apart (hexagonal grid).
    private static readonly float[] DirCos = [1f, -0.5f, -0.5f];
    private static readonly float[] DirSin = [0f, 0.8660254f, -0.8660254f];

    public int CodeDim => _modules * _cellsPerModule;

    public GridCellEncoder(
        int modules, int cellsPerModule,
        float baseFrequency, float frequencyRatio, Random rng)
    {
        if (modules < 1 || cellsPerModule < 1)
            throw new ArgumentOutOfRangeException(nameof(modules), "modules and cellsPerModule must be >= 1.");

        _modules        = modules;
        _cellsPerModule = cellsPerModule;

        _moduleFreq = new float[modules];
        for (int m = 0; m < modules; m++)
            _moduleFreq[m] = baseFrequency * MathF.Pow(frequencyRatio, m);

        // Distribute phases on a regular grid inside each module's unit cell.
        _phaseX = new float[CodeDim];
        _phaseY = new float[CodeDim];
        int side = Math.Max(1, (int)MathF.Round(MathF.Sqrt(cellsPerModule)));
        for (int m = 0; m < modules; m++)
        {
            float period = 2f * MathF.PI / _moduleFreq[m];
            for (int k = 0; k < cellsPerModule; k++)
            {
                int idx = m * cellsPerModule + k;
                int gx  = k % side, gy = k / side;
                // jitter keeps phases from perfectly overlapping when cellsPerModule isn't square
                _phaseX[idx] = (gx + 0.5f) / side * period + (float)rng.NextDouble() * 0.01f;
                _phaseY[idx] = (gy + 0.5f) / side * period + (float)rng.NextDouble() * 0.01f;
            }
        }
    }

    /// <summary>
    /// Returns the rectified grid-cell activation vector for position (x, y).
    /// Values lie in [0, 1]; each cell peaks on a hexagonal lattice of locations.
    /// </summary>
    public DenseTensor<float> Encode(float x, float y)
    {
        var code = new DenseTensor<float>(CodeDim);
        for (int m = 0; m < _modules; m++)
        {
            float f = _moduleFreq[m];
            for (int k = 0; k < _cellsPerModule; k++)
            {
                int   idx = m * _cellsPerModule + k;
                float px  = _phaseX[idx], py = _phaseY[idx];
                float g   = 0f;
                for (int d = 0; d < 3; d++)
                    g += MathF.Cos(f * (DirCos[d] * (x - px) + DirSin[d] * (y - py)));
                g /= 3f;                       // g ∈ [-0.5, 1]
                code.Buffer[idx] = MathF.Max(0f, g);   // rectify → sparse grid fields
            }
        }
        return code;
    }
}
