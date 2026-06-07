using PredictiveCoding.Core;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// FrontalEyeFields  —  saccade selection / overt attention (active vision)
//
// The frontal eye fields and superior colliculus decide where to look next: they
// build a priority (saliency) map over the visual field and select the next
// fixation, suppressing already-visited locations (inhibition of return).  This is
// the overt-attention complement to VisualCortex perception: it drives the
// RetinotopicMap's receptive fields, choosing which patch to foveate.  A bottom-up
// saliency = per-patch contrast (variance); the winner-take-all + IOR scan is the
// Itti–Koch policy, an approximation to active-inference fixation selection.
//
// References: Schall (2004) FEF; Itti & Koch (2001) saliency; Mirza et al. (2016)
//             active inference of saccades.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class FrontalEyeFields
{
    private readonly RetinotopicMap _map;

    public RetinotopicMap Retinotopy => _map;
    public int PatchCount => _map.PatchCount;
    public int GridHeight => _map.GridHeight;
    public int GridWidth  => _map.GridWidth;

    public FrontalEyeFields(int channels, int height, int width, int patchSize, int stride)
        => _map = new RetinotopicMap(channels, height, width, patchSize, stride);

    /// <summary>
    /// Bottom-up saliency per receptive field: the variance (local contrast) of each
    /// retinotopic patch — high-contrast regions are more salient.  One value per
    /// patch, row-major over the grid.
    /// </summary>
    public float[] SaliencyMap(DenseTensor<float> image)
    {
        var patches = _map.Extract(image);
        var s = new float[patches.Length];
        for (int p = 0; p < patches.Length; p++) s[p] = Variance(patches[p].Buffer);
        return s;
    }

    /// <summary>
    /// Selects the next fixation: the most salient patch not in <paramref name="visited"/>
    /// (inhibition of return).  Returns the patch index, or −1 when all are visited.
    /// </summary>
    public int NextSaccade(DenseTensor<float> image, IReadOnlySet<int> visited)
    {
        var s = SaliencyMap(image);
        return ArgMaxUnvisited(s, visited);
    }

    /// <summary>
    /// A scan path of up to <paramref name="fixations"/> patch indices, greedily
    /// foveating the most salient unvisited location each step (saliency + inhibition
    /// of return — the Itti–Koch winner-take-all scan).
    /// </summary>
    public List<int> Scan(DenseTensor<float> image, int fixations)
    {
        if (fixations < 1) throw new ArgumentOutOfRangeException(nameof(fixations));
        var s       = SaliencyMap(image);
        var visited = new HashSet<int>();
        var path    = new List<int>(Math.Min(fixations, s.Length));
        for (int f = 0; f < fixations; f++)
        {
            int best = ArgMaxUnvisited(s, visited);
            if (best < 0) break;
            visited.Add(best);
            path.Add(best);
        }
        return path;
    }

    /// <summary>(row, col) grid position of a patch index.</summary>
    public (int row, int col) PatchPosition(int patchIndex)
        => (patchIndex / _map.GridWidth, patchIndex % _map.GridWidth);

    private static int ArgMaxUnvisited(float[] s, IReadOnlySet<int> visited)
    {
        int best = -1; float bestSal = float.NegativeInfinity;
        for (int p = 0; p < s.Length; p++)
        {
            if (visited.Contains(p)) continue;
            if (s[p] > bestSal) { bestSal = s[p]; best = p; }
        }
        return best;
    }

    private static float Variance(float[] v)
    {
        float mean = 0f;
        for (int i = 0; i < v.Length; i++) mean += v[i];
        mean /= v.Length;
        float var = 0f;
        for (int i = 0; i < v.Length; i++) { float d = v[i] - mean; var += d * d; }
        return var / v.Length;
    }
}
