using PredictiveCoding.Networks;

namespace PredictiveCoding.Expanded.Graph;

// ─────────────────────────────────────────────────────────────────────────────
// GraphPredictiveNetworkConfig  —  configuration for a Graph Predictive Coding net
//
// A single shared "edge predictor" PCN learns the relational map that turns a
// node's feature vector into a prediction of its neighbours' feature vectors.
// Inference over the graph is message passing by prediction-error minimisation:
// each node is nudged toward the consensus of its neighbours' predictions.  This
// extends predictive coding from sequences to arbitrary graph structure (scenes,
// meshes, molecules, social networks).
//
//   EdgePredictorConfig.LayerSizes[0]  = node feature dimensionality
//   EdgePredictorConfig.LayerSizes[^1] = node feature dimensionality (same)
//
// References:
//   Rao & Ballard (1999) — predictive coding.
//   Gklezakos & Rao (2022) "Active Predictive Coding Networks" — structured PC.
//   Scarselli et al. (2009) GNN — relational message-passing analogue.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record GraphPredictiveNetworkConfig
{
    /// <summary>
    /// Shared edge-predictor PC topology.  Maps a node feature to a predicted
    /// neighbour feature, so LayerSizes[0] and LayerSizes[^1] must both equal the
    /// node feature dimensionality.
    /// </summary>
    public NetworkConfig EdgePredictorConfig { get; init; } = new();

    /// <summary>
    /// Blend rate α ∈ (0, 1]: per round, an unobserved node moves
    ///   μ_i ← (1 − α)·μ_i + α·(neighbour consensus).
    /// Smaller values give smoother, slower propagation.
    /// </summary>
    public float BlendRate { get; init; } = 0.5f;

    public string BasePath { get; init; } = "./brain/graph";

    public void Validate()
    {
        int inDim  = EdgePredictorConfig.LayerSizes[0];
        int outDim = EdgePredictorConfig.LayerSizes[^1];
        if (inDim != outDim)
            throw new InvalidOperationException(
                $"EdgePredictorConfig input ({inDim}) and output ({outDim}) dims must match " +
                "(both equal the node feature dimensionality).");
        if (BlendRate <= 0f || BlendRate > 1f)
            throw new InvalidOperationException("BlendRate must be in (0, 1].");
    }
}
