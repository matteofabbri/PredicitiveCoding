using PredictiveCoding.Core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Networks.Graph;

// ─────────────────────────────────────────────────────────────────────────────
// GraphPredictiveNetwork  —  predictive coding over arbitrary graph structure
//
// Every node owns a feature vector; a single shared "edge predictor" PCN learns
// the relational map source-feature → neighbour-feature.  Training presents each
// directed edge (i → j) as a supervised pair (feature_i, feature_j), so the model
// learns "given a node, predict what its neighbours look like".
//
// Inference is message passing by prediction-error minimisation: to fill in or
// denoise a node, gather every neighbour's prediction of it and blend the node
// toward that consensus, iterating for a few rounds.  Observed nodes stay fixed
// and act as boundary conditions, exactly as clamped layers do in a chain PCN.
//
// References:
//   Rao & Ballard (1999); Friston (2005) — predictive coding.
//   Scarselli et al. (2009) — graph neural networks (message passing analogue).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class GraphPredictiveNetwork : IDisposable
{
    private readonly Network                      _edgePredictor;
    private readonly GraphPredictiveNetworkConfig _config;

    /// <summary>Node feature dimensionality.</summary>
    public int FeatureDim => _edgePredictor.Configuration.LayerSizes[0];

    public GraphPredictiveNetworkConfig Configuration => _config;

    public GraphPredictiveNetwork(GraphPredictiveNetworkConfig config, Random rng)
    {
        config.Validate();
        _config = config;
        var netCfg = config.EdgePredictorConfig with { Path = config.BasePath };
        _edgePredictor = new Network(netCfg, rng);
    }

    // ── Learning ─────────────────────────────────────────────────────────────

    /// <summary>
    /// One learning pass over the graph: for every directed edge (i → j) the
    /// shared edge predictor learns to map node i's feature to node j's feature.
    /// <paramref name="neighbours"/>[i] lists the indices of node i's neighbours.
    /// </summary>
    public void Learn(
        IReadOnlyList<DenseTensor<float>> nodeFeatures,
        IReadOnlyList<int[]>              neighbours,
        int                              inferenceSteps)
    {
        ValidateGraph(nodeFeatures, neighbours);

        for (int i = 0; i < nodeFeatures.Count; i++)
            foreach (int j in neighbours[i])
                _edgePredictor.Learn(nodeFeatures[i], nodeFeatures[j], inferenceSteps);
    }

    // ── Inference / completion ──────────────────────────────────────────────────

    /// <summary>
    /// Completes / denoises node features by iterative message passing.  Nodes
    /// flagged in <paramref name="observed"/> are held fixed (boundary
    /// conditions); every other node is repeatedly blended toward the mean of its
    /// neighbours' predictions of it.  Returns the reconstructed feature per node.
    /// </summary>
    public DenseTensor<float>[] Complete(
        IReadOnlyList<DenseTensor<float>> nodeFeatures,
        IReadOnlyList<int[]>              neighbours,
        IReadOnlyList<bool>              observed,
        int                              inferenceSteps,
        int                              rounds)
    {
        ValidateGraph(nodeFeatures, neighbours);
        if (observed.Count != nodeFeatures.Count)
            throw new ArgumentException("observed must have one flag per node.");
        if (rounds < 1) throw new ArgumentOutOfRangeException(nameof(rounds), "rounds must be >= 1.");

        int n = nodeFeatures.Count;
        var cur = new DenseTensor<float>[n];
        for (int i = 0; i < n; i++) cur[i] = nodeFeatures[i].Clone();

        float alpha = _config.BlendRate;

        for (int r = 0; r < rounds; r++)
        {
            // Predict every node from the current estimates, then commit together
            // so updates within a round do not see each other (Jacobi-style).
            var next = new DenseTensor<float>[n];
            for (int i = 0; i < n; i++)
            {
                if (observed[i] || neighbours[i].Length == 0)
                {
                    next[i] = cur[i];
                    continue;
                }

                // Consensus = mean over neighbours j of edgePredictor(cur[j]),
                // i.e. each neighbour's prediction of node i.
                var consensus = new DenseTensor<float>(FeatureDim);
                foreach (int j in neighbours[i])
                {
                    var predOfI = _edgePredictor.InferGetTop(cur[j], inferenceSteps);
                    TensorUtils.InPlaceAxpy(1f, predOfI, consensus);
                }
                float inv = 1f / neighbours[i].Length;
                for (int d = 0; d < FeatureDim; d++)
                    consensus.Buffer[d] *= inv;

                var blended = cur[i].Clone();
                for (int d = 0; d < FeatureDim; d++)
                    blended.Buffer[d] = (1f - alpha) * blended.Buffer[d] + alpha * consensus.Buffer[d];
                next[i] = blended;
            }
            cur = next;
        }

        return cur;
    }

    // ── Diagnostics ──────────────────────────────────────────────────────────

    /// <summary>
    /// Total graph prediction error: Σ over edges (i → j) of
    /// ½‖feature_j − edgePredictor(feature_i)‖².  Lower means the relational model
    /// explains the graph better.
    /// </summary>
    public float GraphEnergy(
        IReadOnlyList<DenseTensor<float>> nodeFeatures,
        IReadOnlyList<int[]>              neighbours,
        int                              inferenceSteps)
    {
        ValidateGraph(nodeFeatures, neighbours);

        float energy = 0f;
        for (int i = 0; i < nodeFeatures.Count; i++)
            foreach (int j in neighbours[i])
            {
                var pred = _edgePredictor.InferGetTop(nodeFeatures[i], inferenceSteps);
                energy += 0.5f * TensorUtils.SumSquares(TensorUtils.ElementwiseSub(nodeFeatures[j], pred));
            }
        return energy;
    }

    public void FlushStore() => _edgePredictor.FlushStore();

    // ── Private helpers ───────────────────────────────────────────────────────

    private void ValidateGraph(
        IReadOnlyList<DenseTensor<float>> nodeFeatures,
        IReadOnlyList<int[]>              neighbours)
    {
        if (nodeFeatures.Count != neighbours.Count)
            throw new ArgumentException("nodeFeatures and neighbours must have the same length.");
        for (int i = 0; i < nodeFeatures.Count; i++)
        {
            if (nodeFeatures[i].Length != FeatureDim)
                throw new ArgumentException(
                    $"Node {i} feature has {nodeFeatures[i].Length} elements; expected {FeatureDim}.");
            foreach (int j in neighbours[i])
                if (j < 0 || j >= nodeFeatures.Count)
                    throw new ArgumentException($"Node {i} has out-of-range neighbour index {j}.");
        }
    }

    // ── Disposal ─────────────────────────────────────────────────────────────

    public void Dispose() => _edgePredictor.Dispose();
}
