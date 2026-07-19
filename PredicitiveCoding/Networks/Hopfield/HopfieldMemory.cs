using PredictiveCoding.Core;

namespace PredictiveCoding.Networks.Hopfield;

// ─────────────────────────────────────────────────────────────────────────────
// HopfieldMemory  —  modern (continuous) Hopfield associative store
//
// Stores a set of d-dimensional patterns and retrieves them by the modern
// Hopfield update rule (Ramsauer et al. 2021):
//
//   ξ_{t+1} = X · softmax(β · Xᵀ ξ_t)
//
// where the columns of X are the stored patterns.  A single update already
// pulls the query toward the most similar stored pattern; iterating a few times
// drives it into the nearest attractor, giving exponential storage capacity and
// one-/few-step retrieval.  This is the associative-memory component that
// stabilises the latent states produced by a predictive-coding network.
//
// Reference:
//   Ramsauer et al. (2021) "Hopfield Networks is All You Need", ICLR.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class HopfieldMemory
{
    private readonly List<float[]> _patterns = [];
    private readonly int           _dim;

    public HopfieldMemory(int dim)
    {
        if (dim < 1) throw new ArgumentOutOfRangeException(nameof(dim), "dim must be >= 1.");
        _dim = dim;
    }

    /// <summary>Number of stored patterns.</summary>
    public int Count => _patterns.Count;

    /// <summary>Dimensionality of every stored pattern.</summary>
    public int Dim => _dim;

    /// <summary>Adds a pattern to the store (kept by value).</summary>
    public void Store(DenseTensor<float> pattern)
    {
        if (pattern.Length != _dim)
            throw new ArgumentException($"pattern has {pattern.Length} elements; expected {_dim}.");
        _patterns.Add(pattern.Buffer.ToArray());
    }

    /// <summary>Removes all stored patterns.</summary>
    public void Clear() => _patterns.Clear();

    /// <summary>
    /// Retrieves the attractor nearest to <paramref name="query"/> by running
    /// <paramref name="iterations"/> modern-Hopfield updates with inverse
    /// temperature <paramref name="beta"/>.  Returns a clone of the query when the
    /// store is empty.
    /// </summary>
    public DenseTensor<float> Retrieve(DenseTensor<float> query, float beta, int iterations)
    {
        if (query.Length != _dim)
            throw new ArgumentException($"query has {query.Length} elements; expected {_dim}.");
        if (_patterns.Count == 0) return query.Clone();

        var xi = query.Buffer.ToArray();

        for (int it = 0; it < iterations; it++)
        {
            // scores_i = β · ⟨X_i, ξ⟩
            var scores = new float[_patterns.Count];
            float max  = float.NegativeInfinity;
            for (int i = 0; i < _patterns.Count; i++)
            {
                float dot = 0f;
                var p = _patterns[i];
                for (int d = 0; d < _dim; d++) dot += p[d] * xi[d];
                scores[i] = beta * dot;
                if (scores[i] > max) max = scores[i];
            }

            // softmax
            float sum = 0f;
            for (int i = 0; i < scores.Length; i++)
            {
                scores[i] = MathF.Exp(scores[i] - max);
                sum += scores[i];
            }

            // ξ ← Σ_i w_i · X_i
            var next = new float[_dim];
            for (int i = 0; i < _patterns.Count; i++)
            {
                float w = scores[i] / sum;
                var   p = _patterns[i];
                for (int d = 0; d < _dim; d++) next[d] += w * p[d];
            }
            xi = next;
        }

        var result = new DenseTensor<float>(_dim);
        xi.CopyTo(result.Buffer, 0);
        return result;
    }
}
