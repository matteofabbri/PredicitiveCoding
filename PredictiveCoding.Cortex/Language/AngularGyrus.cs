using PredictiveCoding.Core;
using PredictiveCoding.Expanded.Graph;

namespace PredictiveCoding.Cortex;

// ─────────────────────────────────────────────────────────────────────────────
// AngularGyrus  —  cross-modal semantic hub (relational concept graph)
//
// The angular gyrus is a heteromodal hub that binds concepts into a relational
// semantic network: it supports inferring an associated concept from related ones
// (e.g. word ↔ meaning ↔ referent).  Implemented over a GraphPredictiveNetwork:
// concepts are nodes, semantic relations are edges, and a shared edge predictor
// learns "given a concept, predict its neighbours", completing missing concepts by
// message passing.
//
// References: Binder & Desai (2011) semantic system; Seghier (2013) angular gyrus.
// ─────────────────────────────────────────────────────────────────────────────

public sealed record AngularGyrusConfig
{
    /// <summary>Relational edge predictor over concept features (input dim == output dim).</summary>
    public GraphPredictiveNetworkConfig Relations { get; init; } = new();

    public string BasePath { get; init; } = "./cortex/angular";

    public void Validate() => Relations.Validate();
}

public sealed class AngularGyrus : IDisposable
{
    private readonly GraphPredictiveNetwork _graph;

    public int ConceptDim => _graph.FeatureDim;

    public AngularGyrus(AngularGyrusConfig config, Random rng)
    {
        config.Validate();
        _graph = new GraphPredictiveNetwork(config.Relations with { BasePath = config.BasePath }, rng);
    }

    /// <summary>Learns the relations: for each concept, predict the concepts it relates to.</summary>
    public void Learn(IReadOnlyList<DenseTensor<float>> concepts, IReadOnlyList<int[]> relations, int steps)
        => _graph.Learn(concepts, relations, steps);

    /// <summary>
    /// Completes / denoises concepts from their related ones by message passing.
    /// Concepts flagged in <paramref name="observed"/> are held fixed.
    /// </summary>
    public DenseTensor<float>[] Associate(
        IReadOnlyList<DenseTensor<float>> concepts,
        IReadOnlyList<int[]>              relations,
        IReadOnlyList<bool>              observed,
        int steps, int rounds)
        => _graph.Complete(concepts, relations, observed, steps, rounds);

    /// <summary>Total relational prediction error — how well the semantic graph is explained.</summary>
    public float RelationalEnergy(IReadOnlyList<DenseTensor<float>> concepts, IReadOnlyList<int[]> relations, int steps)
        => _graph.GraphEnergy(concepts, relations, steps);

    public void FlushAll() => _graph.FlushStore();
    public void Dispose()  => _graph.Dispose();
}
