namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// SuperiorColliculus  —  multisensory orienting / gaze-and-head priority map
//
// The superior colliculus builds a spatial priority map that fuses visual, auditory
// (and tactile) saliency over a common retinotopic frame and selects the next
// orienting movement toward the most salient location — with inhibition of return so
// the scan moves on. Its hallmark is *multisensory enhancement*: a stimulus seen and
// heard at the same place is more salient than either alone (super-additive). It is
// the orienting complement to `FrontalEyeFields` (cortical saccade selection).
//
// References: Stein & Meredith (1993) multisensory integration in SC;
//             Sparks (1986) collicular motor map.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class SuperiorColliculus
{
    /// <summary>Number of spatial cells in the (shared) priority map.</summary>
    public int CellCount { get; }

    public float VisualWeight { get; }
    public float AuditoryWeight { get; }

    /// <summary>Super-additive bonus when a location is salient in both modalities
    /// (0 = plain weighted sum).</summary>
    public float MultisensoryEnhancement { get; }

    public SuperiorColliculus(int cellCount, float visualWeight = 1f, float auditoryWeight = 1f,
        float multisensoryEnhancement = 0f)
    {
        if (cellCount < 1) throw new ArgumentOutOfRangeException(nameof(cellCount));
        if (visualWeight < 0f) throw new ArgumentOutOfRangeException(nameof(visualWeight));
        if (auditoryWeight < 0f) throw new ArgumentOutOfRangeException(nameof(auditoryWeight));
        if (multisensoryEnhancement < 0f) throw new ArgumentOutOfRangeException(nameof(multisensoryEnhancement));
        CellCount = cellCount;
        VisualWeight = visualWeight;
        AuditoryWeight = auditoryWeight;
        MultisensoryEnhancement = multisensoryEnhancement;
    }

    /// <summary>Fuses the visual and auditory saliency maps (same length = CellCount)
    /// into one priority map, with optional super-additive multisensory enhancement.</summary>
    public float[] PriorityMap(IReadOnlyList<float> visual, IReadOnlyList<float> auditory)
    {
        ArgumentNullException.ThrowIfNull(visual);
        ArgumentNullException.ThrowIfNull(auditory);
        if (visual.Count != CellCount || auditory.Count != CellCount)
            throw new ArgumentException($"Both maps must have {CellCount} cells.");

        var p = new float[CellCount];
        for (int i = 0; i < CellCount; i++)
        {
            float v = visual[i], a = auditory[i];
            p[i] = VisualWeight * v + AuditoryWeight * a
                 + MultisensoryEnhancement * MathF.Max(0f, v) * MathF.Max(0f, a);
        }
        return p;
    }

    /// <summary>Selects the highest-priority cell not in <paramref name="visited"/>
    /// (orienting target + inhibition of return). −1 when all are visited.</summary>
    public int Orient(IReadOnlyList<float> visual, IReadOnlyList<float> auditory, IReadOnlySet<int> visited)
        => ArgMaxUnvisited(PriorityMap(visual, auditory), visited);

    /// <summary>A scan path of up to <paramref name="targets"/> orienting moves, each
    /// to the most salient unvisited cell (winner-take-all + inhibition of return).</summary>
    public List<int> Scan(IReadOnlyList<float> visual, IReadOnlyList<float> auditory, int targets)
    {
        if (targets < 1) throw new ArgumentOutOfRangeException(nameof(targets));
        var p = PriorityMap(visual, auditory);
        var visited = new HashSet<int>();
        var path = new List<int>(Math.Min(targets, CellCount));
        for (int t = 0; t < targets; t++)
        {
            int best = ArgMaxUnvisited(p, visited);
            if (best < 0) break;
            visited.Add(best);
            path.Add(best);
        }
        return path;
    }

    private static int ArgMaxUnvisited(float[] priority, IReadOnlySet<int> visited)
    {
        int best = -1;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < priority.Length; i++)
        {
            if (visited.Contains(i)) continue;
            if (priority[i] > bestVal) { bestVal = priority[i]; best = i; }
        }
        return best;
    }
}
