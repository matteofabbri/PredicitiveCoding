using PredictiveCoding.Core;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// Claustrum  —  global binding / workspace synchronisation ("the conductor")
//
// The claustrum is reciprocally connected with almost the entire cortex and is
// proposed to coordinate / synchronise activity across areas — a global-workspace
// conductor.  This pure controller computes a consensus latent (the mean of the
// areas' latents) to broadcast back to every area, and a coherence score (mean
// pairwise cosine) reporting how "bound"/ignited the workspace currently is.
//
// References: Crick & Koch (2005) function of the claustrum; Baars (1988) global workspace.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class Claustrum
{
    /// <summary>
    /// Global broadcast: the element-wise mean of the area latents — the consensus
    /// signal sent back to every area.  All latents must share a dimension.
    /// </summary>
    public DenseTensor<float> Synchronize(IReadOnlyList<DenseTensor<float>> areaLatents)
    {
        int n   = Count(areaLatents);
        int dim = areaLatents[0].Length;
        var mean = new DenseTensor<float>(dim);
        for (int a = 0; a < n; a++)
        {
            var v = areaLatents[a];
            if (v.Length != dim)
                throw new ArgumentException("All area latents must share the same dimension.");
            for (int d = 0; d < dim; d++) mean.Buffer[d] += v.Buffer[d];
        }
        float inv = 1f / n;
        for (int d = 0; d < dim; d++) mean.Buffer[d] *= inv;
        return mean;
    }

    /// <summary>
    /// Workspace coherence ∈ [−1, 1]: the mean pairwise cosine similarity between area
    /// latents — how aligned/bound the global workspace is (1 = all areas agree).
    /// Returns 1 for a single area.
    /// </summary>
    public float Coherence(IReadOnlyList<DenseTensor<float>> areaLatents)
    {
        int n = Count(areaLatents);
        if (n == 1) return 1f;
        int dim = areaLatents[0].Length;

        float sum = 0f; int pairs = 0;
        for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
        {
            if (areaLatents[i].Length != dim || areaLatents[j].Length != dim)
                throw new ArgumentException("All area latents must share the same dimension.");
            sum += Cosine(areaLatents[i].Buffer, areaLatents[j].Buffer);
            pairs++;
        }
        return sum / pairs;
    }

    private static int Count(IReadOnlyList<DenseTensor<float>> latents)
    {
        if (latents is null || latents.Count == 0)
            throw new ArgumentException("At least one area latent is required.", nameof(latents));
        return latents.Count;
    }

    private static float Cosine(float[] a, float[] b)
    {
        float dot = 0f, na = 0f, nb = 0f;
        for (int i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        float denom = MathF.Sqrt(na) * MathF.Sqrt(nb);
        return denom < 1e-9f ? 0f : dot / denom;
    }
}
