using PredictiveCoding.Core;
using PredictiveCoding.Cortex;

namespace PredictiveCoding.CortexApp.Examples;

// Renders the spatial firing maps of a few entorhinal grid cells over a 2-D arena.
public static class GridCellsExample
{
    public sealed record Result(DenseTensor<float>[] Maps, int[] CellIndices, int Resolution);

    public static Result Run(int resolution)
    {
        var enc    = new GridCellEncoder(modules: 4, cellsPerModule: 16,
                                         baseFrequency: 0.6f, frequencyRatio: 1.5f, new Random(11));
        var cells  = new[] { 2, 22, 42 };   // cells drawn from three different modules
        float extent = 14f;                  // arena spans several grid periods

        var maps = new DenseTensor<float>[cells.Length];
        for (int m = 0; m < cells.Length; m++)
        {
            var map = new DenseTensor<float>(1, resolution, resolution);
            for (int iy = 0; iy < resolution; iy++)
            for (int ix = 0; ix < resolution; ix++)
            {
                float x = ix / (float)(resolution - 1) * extent;
                float y = iy / (float)(resolution - 1) * extent;
                map[0, iy, ix] = enc.Encode(x, y).Buffer[cells[m]];
            }
            maps[m] = map;
        }
        return new Result(maps, cells, resolution);
    }
}
