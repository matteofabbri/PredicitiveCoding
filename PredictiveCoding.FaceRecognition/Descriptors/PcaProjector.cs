namespace PredictiveCoding.FaceRecognition.Descriptors;

/// <summary>
/// Truncated PCA with whitening via <b>subspace power iteration</b>.
/// <para>
/// Algorithm: Simultaneous Iteration (power method on the sample-covariance eigenvectors).
/// Complexity per iteration: O(n · d · k) — no explicit d×d covariance matrix is ever formed,
/// so it scales to the high-dimensional descriptors produced by LBP and HOG.
/// </para>
/// <para>
/// After <see cref="Fit"/>, <see cref="Project"/> maps a d-dimensional descriptor to a
/// whitened k-dimensional vector that can be compared with cosine distance.
/// Whitening (dividing each PC by its standard deviation) equalises the contribution of
/// all components and significantly improves verification accuracy.
/// </para>
/// </summary>
public sealed class PcaProjector
{
    // Stored after Fit: _components[j * d + p] = p-th element of j-th eigenvector (row-major)
    private float[]? _mean;
    private float[]? _components;
    private float[]? _scales;   // 1/sqrt(eigenvalue_j) for whitening

    public int InputDim  { get; private set; }
    public int OutputDim { get; private set; }
    public bool IsFit    => _components is not null;

    /// <summary>
    /// Fits the PCA projector on the given descriptors.
    /// </summary>
    /// <param name="data">Training vectors (all the same length d).</param>
    /// <param name="targetDim">Number of principal components to keep.</param>
    /// <param name="maxSamples">Maximum training vectors used (cap for speed).</param>
    /// <param name="iterations">Power iteration steps (10–20 is usually enough).</param>
    /// <param name="onProgress">Optional progress callback.</param>
    public void Fit(
        IReadOnlyList<float[]> data,
        int             targetDim,
        int             maxSamples = 2000,
        int             iterations = 12,
        Action<string>? onProgress = null,
        Random?         rng        = null)
    {
        rng ??= new Random(17);

        // Cap to maxSamples (random subsample)
        IReadOnlyList<float[]> subset = data.Count <= maxSamples
            ? data
            : Subsample(data, maxSamples, rng);

        int n = subset.Count;
        int d = subset[0].Length;
        int k = Math.Min(targetDim, Math.Min(n - 1, d));

        InputDim  = d;
        OutputDim = k;

        onProgress?.Invoke($"PCA: {n} samples, {d}-D → {k}-D  ({iterations} iterations) …");

        // ── Mean ──────────────────────────────────────────────────────────────────
        _mean = new float[d];
        for (int i = 0; i < n; i++)
        {
            float[] xi = subset[i];
            for (int p = 0; p < d; p++) _mean[p] += xi[p];
        }
        for (int p = 0; p < d; p++) _mean[p] /= n;

        // ── Random initialisation of the subspace V (rows = eigenvectors) ────────
        // V[j * d + p] = p-th coordinate of j-th basis vector (row-major over k)
        var V = new float[k * d];
        for (int j = 0; j < k; j++)
            for (int p = 0; p < d; p++)
                V[j * d + p] = (float)(rng.NextDouble() * 2.0 - 1.0);
        ModifiedGramSchmidt(V, k, d);

        var Z = new float[n * k];  // Z[i * k + j] = projection of sample i onto PC j
        var W = new float[k * d];  // W[j * d + p] = next iteration V (A V result)

        // ── Power iterations ───────────────────────────────────────────────────────
        for (int t = 0; t < iterations; t++)
        {
            // Z = X_c V  (n × k)   -- each row: projection of one sample onto k PCs
            Parallel.For(0, n, i =>
            {
                float[] xi = subset[i];
                int     ik = i * k;
                for (int j = 0; j < k; j++)
                {
                    float dot = 0f;
                    int   jd  = j * d;
                    for (int p = 0; p < d; p++)
                        dot += (xi[p] - _mean[p]) * V[jd + p];
                    Z[ik + j] = dot;
                }
            });

            // W = X_c^T Z  (d × k as k × d, row j = covariance × basis j)
            Array.Clear(W);
            for (int i = 0; i < n; i++)
            {
                float[] xi = subset[i];
                int     ik = i * k;
                for (int j = 0; j < k; j++)
                {
                    float zij = Z[ik + j];
                    if (zij == 0f) continue;
                    int jd = j * d;
                    for (int p = 0; p < d; p++)
                        W[jd + p] += (xi[p] - _mean[p]) * zij;
                }
            }

            // V ← QR(W)
            ModifiedGramSchmidt(W, k, d);
            Buffer.BlockCopy(W, 0, V, 0, V.Length * sizeof(float));

            onProgress?.Invoke($"  iter {t + 1}/{iterations}");
        }

        // ── Eigenvalues from final Z ───────────────────────────────────────────────
        // Recompute Z = X_c V with converged V
        Parallel.For(0, n, i =>
        {
            float[] xi = subset[i];
            int     ik = i * k;
            for (int j = 0; j < k; j++)
            {
                float dot = 0f;
                int   jd  = j * d;
                for (int p = 0; p < d; p++)
                    dot += (xi[p] - _mean[p]) * V[jd + p];
                Z[ik + j] = dot;
            }
        });

        var eigenvalues = new float[k];
        for (int j = 0; j < k; j++)
        {
            float s = 0f;
            for (int i = 0; i < n; i++)
                s += Z[i * k + j] * Z[i * k + j];
            eigenvalues[j] = s / n;
        }

        // ── Sort descending by eigenvalue ──────────────────────────────────────────
        int[] order = Enumerable.Range(0, k).OrderByDescending(j => eigenvalues[j]).ToArray();
        _components = new float[k * d];
        _scales     = new float[k];
        for (int idx = 0; idx < k; idx++)
        {
            int j = order[idx];
            Array.Copy(V, j * d, _components, idx * d, d);
            _scales[idx] = eigenvalues[j] > 1e-8f ? 1f / MathF.Sqrt(eigenvalues[j]) : 0f;
        }

        onProgress?.Invoke(
            $"PCA done. λ_1={eigenvalues[order[0]]:F2}, λ_{k}={eigenvalues[order[k - 1]]:F4}");
    }

    /// <summary>
    /// Projects <paramref name="x"/> into the whitened PCA space and L2-normalises the result.
    /// </summary>
    public float[] Project(float[] x)
    {
        if (_components is null)
            throw new InvalidOperationException("Call Fit before Project.");

        var y = new float[OutputDim];
        for (int j = 0; j < OutputDim; j++)
        {
            float dot = 0f;
            int   jd  = j * InputDim;
            for (int p = 0; p < InputDim; p++)
                dot += (x[p] - _mean![p]) * _components[jd + p];
            y[j] = dot * _scales![j];
        }

        // L2 normalise for cosine distance
        float norm = 0f;
        for (int j = 0; j < OutputDim; j++) norm += y[j] * y[j];
        norm = MathF.Sqrt(norm);
        if (norm > 1e-8f)
            for (int j = 0; j < OutputDim; j++) y[j] /= norm;

        return y;
    }

    // ── Helpers ───────────────────────────────────────────────────────────────────

    /// <summary>Modified Gram-Schmidt in-place orthonormalisation of the rows of A (each row length <paramref name="cols"/>).</summary>
    private static void ModifiedGramSchmidt(float[] A, int rows, int cols)
    {
        for (int j = 0; j < rows; j++)
        {
            int jd = j * cols;
            for (int i = 0; i < j; i++)
            {
                int   id  = i * cols;
                float dot = 0f;
                for (int p = 0; p < cols; p++) dot += A[jd + p] * A[id + p];
                for (int p = 0; p < cols; p++) A[jd + p] -= dot * A[id + p];
            }
            float norm = 0f;
            for (int p = 0; p < cols; p++) norm += A[jd + p] * A[jd + p];
            norm = MathF.Sqrt(norm);
            if (norm > 1e-8f)
                for (int p = 0; p < cols; p++) A[jd + p] /= norm;
        }
    }

    private static List<float[]> Subsample(IReadOnlyList<float[]> data, int n, Random rng)
    {
        var idx = Enumerable.Range(0, data.Count).ToList();
        for (int i = idx.Count - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (idx[i], idx[j]) = (idx[j], idx[i]);
        }
        return idx.Take(n).Select(i => data[i]).ToList();
    }
}
