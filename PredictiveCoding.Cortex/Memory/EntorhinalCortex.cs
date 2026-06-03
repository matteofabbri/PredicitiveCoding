using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// EntorhinalCortex  —  spatial predictive coding with grid-cell codes
//
// The entorhinal cortex supplies the hippocampus with a metric code for space.
// Here a GridCellEncoder turns a 2-D position into a grid-cell population vector,
// and a PCN learns the predictive association between that spatial code and the
// observation experienced at the location.  Querying the code at a position then
// predicts what should be observed there (a spatial generative model).
//
// References: Hafting et al. (2005); Whittington et al. (2020) Tolman-Eichenbaum.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record EntorhinalCortexConfig
{
    public int   GridModules     { get; init; } = 4;
    public int   CellsPerModule  { get; init; } = 16;
    public float BaseFrequency   { get; init; } = 0.5f;
    public float FrequencyRatio  { get; init; } = 1.42f;

    /// <summary>Association PCN. LayerSizes[0] must equal GridModules·CellsPerModule;
    /// LayerSizes[^1] = observation dimensionality predicted at each location.</summary>
    public NetworkConfig Association { get; init; } = new();

    public string BasePath { get; init; } = "./cortex/entorhinal";

    public int CodeDim => GridModules * CellsPerModule;

    public void Validate()
    {
        if (GridModules < 1 || CellsPerModule < 1)
            throw new InvalidOperationException("GridModules and CellsPerModule must be >= 1.");
        if (Association.LayerSizes[0] != CodeDim)
            throw new InvalidOperationException(
                $"Association.LayerSizes[0] ({Association.LayerSizes[0]}) must equal the grid code dim ({CodeDim}).");
    }
}

public sealed class EntorhinalCortex : IDisposable
{
    private readonly GridCellEncoder _grid;
    private readonly Network         _association;

    public int CodeDim        => _grid.CodeDim;
    public int ObservationDim => _association.Configuration.LayerSizes[^1];

    public EntorhinalCortex(EntorhinalCortexConfig config, Random rng)
    {
        config.Validate();
        _grid        = new GridCellEncoder(config.GridModules, config.CellsPerModule,
                                           config.BaseFrequency, config.FrequencyRatio, new Random(rng.Next()));
        _association = new Network(config.Association with { Path = Path.Combine(config.BasePath, "assoc") }, new Random(rng.Next()));
    }

    /// <summary>Grid-cell population code for a position.</summary>
    public DenseTensor<float> EncodePosition(float x, float y) => _grid.Encode(x, y);

    /// <summary>Learns the association "position → observation" for one location.</summary>
    public void Learn(float x, float y, DenseTensor<float> observation, int steps)
        => _association.Learn(_grid.Encode(x, y), observation, steps);

    /// <summary>Predicts the observation expected at a position (spatial generative model).</summary>
    public DenseTensor<float> PredictObservation(float x, float y, int steps)
        => _association.InferGetTop(_grid.Encode(x, y), steps);

    public void FlushAll() => _association.FlushStore();
    public void Dispose()  => _association.Dispose();
}
