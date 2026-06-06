namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// HeterarchicalMessagePassing  —  iterative belief propagation over the cortical graph.
//
// Implements the non-hierarchical message-passing algorithm described in P8:
//
//   Round 0 (initialisation)
//     Each area runs VoteStep on its own sensor input → initial consensus.
//
//   Rounds 1 … MaxRounds
//     For every area:
//       1. Gather the candidate hypothesis sets from all incoming connections,
//          filtered by each connection's Weight threshold.
//       2. Propagate those candidates into the area via PropagateConstraint —
//          this narrows the area's column-level hypothesis sets.
//       3. Re-run VoteStep → updated consensus.
//     If every area has converged (single candidate) → stop early.
//
// Because messages can flow in any direction (FeedForward, FeedBack, Lateral,
// or Heterarchical), even feedback loops are handled naturally; the algorithm
// converges when the belief state reaches a fixed point.
//
// Reference: Hawkins, Leadholm & Clay (2025) P8; Leadholm et al. (2025) P9 §5.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class HeterarchicalMessagePassing
{
    /// <summary>Maximum number of propagation rounds (excluding the initialisation round).</summary>
    public int MaxRounds { get; }

    /// <summary>AgreementScore threshold on the source area's consensus: only object ids
    /// whose score ≥ this value are propagated through each incoming edge, additionally
    /// multiplied by the connection's own Weight.</summary>
    public float ConvergenceThreshold { get; }

    /// <param name="maxRounds">Hard cap on propagation rounds. Default 6.</param>
    /// <param name="convergenceThreshold">Minimum source agreement score before a candidate
    /// is propagated along an edge (before applying the edge weight). Default 0.0 (propagate all).</param>
    public HeterarchicalMessagePassing(int maxRounds = 6, float convergenceThreshold = 0.0f)
    {
        if (maxRounds < 1) throw new ArgumentOutOfRangeException(nameof(maxRounds));
        if (convergenceThreshold < 0f || convergenceThreshold > 1f)
            throw new ArgumentOutOfRangeException(nameof(convergenceThreshold));
        MaxRounds = maxRounds;
        ConvergenceThreshold = convergenceThreshold;
    }

    /// <summary>Runs the full iterative message-passing algorithm on <paramref name="graph"/>.
    /// <paramref name="areaInputs"/> maps each area id to the (feature, location) list for its
    /// columns (length must equal the area's ColumnCount).</summary>
    public IReadOnlyDictionary<int, AreaOutput> PassMessages(
        HeterarchicalGraph graph,
        IReadOnlyDictionary<int, IReadOnlyList<(Sdr Feature, Sdr Location)>> areaInputs)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(areaInputs);

        // Round 0: local inference with no lateral constraints yet.
        var outputs = new Dictionary<int, AreaOutput>();
        foreach (var area in graph.Areas)
        {
            var inputs = RequireInputs(areaInputs, area.AreaId, area.ColumnSystem.ColumnCount);
            area.ColumnSystem.Reset();
            var consensus = area.ColumnSystem.VoteStep(inputs);
            var output = new AreaOutput(area.AreaId, consensus, 0, 0);
            area.UpdateOutput(consensus);
            outputs[area.AreaId] = output;
        }

        // Rounds 1 … MaxRounds: propagate constraints along edges.
        for (int r = 1; r <= MaxRounds; r++)
        {
            bool anyChanged = false;
            foreach (var area in graph.Areas)
            {
                var incoming = IncomingConnections(graph, area.AreaId);
                int messageCount = 0;

                // Collect candidate ids from all incoming connections.
                var externalCandidates = GatherIncomingHypotheses(graph, area.AreaId, incoming);
                if (externalCandidates.Count > 0)
                {
                    area.ColumnSystem.PropagateConstraint(externalCandidates);
                    messageCount = incoming.Count(c =>
                    {
                        var src = graph.GetArea(c.SourceAreaId).CurrentOutput;
                        return src is not null && src.Candidates.Count > 0;
                    });
                }

                var inputs = RequireInputs(areaInputs, area.AreaId, area.ColumnSystem.ColumnCount);
                var consensus = area.ColumnSystem.VoteStep(inputs);

                var prev = outputs[area.AreaId];
                bool changed = !SameSet(prev.Consensus.Candidates, consensus.Candidates);
                anyChanged |= changed;

                var output = new AreaOutput(area.AreaId, consensus, r, messageCount);
                area.UpdateOutput(consensus);
                outputs[area.AreaId] = output;
            }

            if (!anyChanged || IsAllConverged(outputs)) break;
        }

        return outputs;
    }

    /// <summary>Returns the object ids gathered from all source areas connected to
    /// <paramref name="targetAreaId"/>, filtered by each connection's Weight × ConvergenceThreshold.</summary>
    public IReadOnlyList<int> GatherIncomingHypotheses(
        HeterarchicalGraph graph, int targetAreaId)
    {
        var incoming = IncomingConnections(graph, targetAreaId);
        return GatherIncomingHypotheses(graph, targetAreaId, incoming);
    }

    /// <summary>True when every area in <paramref name="outputs"/> has converged to a single candidate.</summary>
    public bool IsAllConverged(IReadOnlyDictionary<int, AreaOutput> outputs)
    {
        ArgumentNullException.ThrowIfNull(outputs);
        return outputs.Values.All(o => o.Consensus.Candidates.Count == 1);
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private IReadOnlyList<int> GatherIncomingHypotheses(
        HeterarchicalGraph graph, int targetAreaId,
        IEnumerable<LongRangeConnection> incoming)
    {
        var candidates = new HashSet<int>();
        foreach (var conn in incoming)
        {
            var source = graph.GetArea(conn.SourceAreaId);
            if (source.CurrentOutput is null) continue;

            float effectiveThreshold = ConvergenceThreshold * conn.Weight;

            // Propagate candidates whose agreement score meets the combined threshold.
            foreach (int id in source.CurrentOutput.Candidates)
                if (source.CurrentOutput.AgreementScore >= effectiveThreshold)
                    candidates.Add(id);
        }
        var result = candidates.ToArray();
        Array.Sort(result);
        return result;
    }

    private static IReadOnlyList<LongRangeConnection> IncomingConnections(
        HeterarchicalGraph graph, int areaId)
        => graph.IncomingConnections(areaId);

    private static IReadOnlyList<(Sdr Feature, Sdr Location)> RequireInputs(
        IReadOnlyDictionary<int, IReadOnlyList<(Sdr Feature, Sdr Location)>> areaInputs,
        int areaId, int expectedCount)
    {
        if (!areaInputs.TryGetValue(areaId, out var inputs))
            throw new KeyNotFoundException(
                $"No inputs provided for area {areaId}.");
        if (inputs.Count != expectedCount)
            throw new ArgumentException(
                $"Area {areaId} has {expectedCount} columns but {inputs.Count} inputs were provided.");
        return inputs;
    }

    private static bool SameSet(IReadOnlyList<int> a, IReadOnlyList<int> b)
    {
        if (a.Count != b.Count) return false;
        var setA = new HashSet<int>(a);
        foreach (int x in b)
            if (!setA.Contains(x)) return false;
        return true;
    }
}
