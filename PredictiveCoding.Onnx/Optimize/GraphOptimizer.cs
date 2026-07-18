using PredictiveCoding.Onnx.Proto;

namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// GraphOptimizer — the save-time transforms behind OnnxModel.Save.
//
// Every transform preserves Run semantics bit-for-bit:
//   • Constant nodes are inlined as initializers.
//   • Constant expressions (all inputs known) are folded by executing the node
//     with the SAME kernels Run would use — the folded value is byte-identical
//     to what the runtime would have computed.
//   • Identity (and inference-mode Dropout) nodes are elided by rewiring names.
//   • Dead nodes and unused initializers are removed (graph outputs decide).
//
// The input graph is never mutated; a transformed copy is returned.
// ─────────────────────────────────────────────────────────────────────────────

public static class GraphOptimizer
{
    // Folding a shape expression yields a handful of numbers; folding something
    // that EXPANDS data (huge constant activations) would bloat the file for no
    // run-time win. Skip folds whose combined outputs exceed this element count.
    private const long FoldedElementCap = 16_000_000;   // ≈ 64 MB of float32

    /// <summary>Returns an optimized copy of <paramref name="graph"/> (same Run results).</summary>
    public static OnnxGraph Optimize(OnnxGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        OnnxGraph g = Clone(graph);
        InlineConstantNodes(g);
        FoldConstantExpressions(g);
        ElideIdentityNodes(g);
        RemoveDeadCode(g);
        return g;
    }

    // Nodes get their name arrays rewritten, so they are copied; attributes,
    // tensors and value infos are never mutated and can be shared.
    private static OnnxGraph Clone(OnnxGraph graph)
    {
        var g = new OnnxGraph
        {
            Name = graph.Name,
            OpsetVersion = graph.OpsetVersion,
            Initializers = new Dictionary<string, OnnxTensorData>(graph.Initializers, StringComparer.Ordinal),
            Inputs = [.. graph.Inputs],
            Outputs = [.. graph.Outputs],
        };
        foreach (OnnxNode n in graph.Nodes)
        {
            g.Nodes.Add(new OnnxNode
            {
                Name = n.Name,
                OpType = n.OpType,
                Inputs = [.. n.Inputs],
                Outputs = [.. n.Outputs],
                Attributes = n.Attributes,
            });
        }
        return g;
    }

    // ── Constant nodes → initializers ───────────────────────────────────────────

    private static void InlineConstantNodes(OnnxGraph g)
    {
        var kept = new List<OnnxNode>(g.Nodes.Count);
        foreach (OnnxNode node in g.Nodes)
        {
            if (node.OpType != "Constant" || node.Outputs.Length != 1 || node.Outputs[0].Length == 0 ||
                ConstantPayload(node) is not OnnxTensorData tensor)
            {
                kept.Add(node);
                continue;
            }
            tensor.Name = node.Outputs[0];
            g.Initializers[node.Outputs[0]] = tensor;
        }
        g.Nodes = kept;
    }

    /// <summary>The tensor a Constant node carries, or null for payload forms we don't support.</summary>
    private static OnnxTensorData? ConstantPayload(OnnxNode node)
    {
        if (node.Attributes.TryGetValue("value", out OnnxAttribute? v) && v.T is not null)
            return new OnnxTensorData
            {
                DataType = v.T.DataType,
                Dims = v.T.Dims,
                Floats = v.T.Floats,
                Longs = v.T.Longs,
            };
        if (node.Attributes.TryGetValue("value_float", out OnnxAttribute? f))
            return new OnnxTensorData { DataType = OnnxDataType.Float, Dims = [], Floats = [f.F] };
        if (node.Attributes.TryGetValue("value_floats", out OnnxAttribute? fs) && fs.Floats is not null)
            return new OnnxTensorData { DataType = OnnxDataType.Float, Dims = [fs.Floats.Length], Floats = fs.Floats };
        if (node.Attributes.TryGetValue("value_int", out OnnxAttribute? i))
            return new OnnxTensorData { DataType = OnnxDataType.Int64, Dims = [], Longs = [i.I] };
        if (node.Attributes.TryGetValue("value_ints", out OnnxAttribute? isAttr) && isAttr.Ints is not null)
            return new OnnxTensorData { DataType = OnnxDataType.Int64, Dims = [isAttr.Ints.Length], Longs = isAttr.Ints };
        return null;   // value_string / sparse_value / …: leave the node alone
    }

    // ── Constant folding ────────────────────────────────────────────────────────

    private static void FoldConstantExpressions(OnnxGraph g)
    {
        // Nodes are topologically sorted, so one forward pass reaches fixpoint:
        // an initializer produced here is visible to every later node.
        var resolved = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        var kept = new List<OnnxNode>(g.Nodes.Count);

        foreach (OnnxNode node in g.Nodes)
        {
            // Zero-input ops (RandomNormal, …) are generators, not constants.
            if (node.Inputs.Length == 0 ||
                !node.Inputs.All(name => name.Length == 0 || g.Initializers.ContainsKey(name)))
            {
                kept.Add(node);
                continue;
            }

            Tensor[] results;
            try
            {
                var args = new Tensor[node.Inputs.Length];
                for (int i = 0; i < args.Length; i++)
                {
                    string name = node.Inputs[i];
                    if (name.Length == 0) continue;   // optional input omitted
                    if (!resolved.TryGetValue(name, out Tensor? t))
                        resolved[name] = t = g.Initializers[name].ToTensor();
                    args[i] = t;
                }
                results = Executor.ExecuteNode(node, args, g.OpsetVersion);
            }
            catch
            {
                // Unsupported op or dtype: leave the node in place. If it sits on
                // an executed path, Run will surface the same error as before.
                kept.Add(node);
                continue;
            }

            // A kernel returning fewer outputs than the node declares would make
            // Run fail loudly; folding would half-populate and mask that error.
            if (results.Length < node.Outputs.Length)
            {
                kept.Add(node);
                continue;
            }

            long totalElements = 0;
            for (int i = 0; i < node.Outputs.Length; i++)
                if (node.Outputs[i].Length > 0)
                    totalElements += results[i].Count;
            if (totalElements > FoldedElementCap)
            {
                kept.Add(node);
                continue;
            }

            for (int i = 0; i < node.Outputs.Length; i++)
            {
                string name = node.Outputs[i];
                if (name.Length == 0) continue;
                g.Initializers[name] = ToTensorData(name, results[i]);
                resolved[name] = results[i];
            }
        }

        g.Nodes = kept;
    }

    // The payload is copied: kernels like Identity/Reshape forward their input
    // array, so storing the result directly could make the folded initializer
    // share memory with the source graph's weights.
    private static OnnxTensorData ToTensorData(string name, Tensor t) => new()
    {
        Name = name,
        DataType = t.IsInt64 ? OnnxDataType.Int64 : OnnxDataType.Float,
        Dims = Array.ConvertAll(t.Shape, d => (long)d),
        Floats = t.IsInt64 ? null : (float[])t.Data.Clone(),
        Longs = (long[]?)t.I64?.Clone(),
    };

    // ── Identity / inference-Dropout elision ────────────────────────────────────

    private static void ElideIdentityNodes(OnnxGraph g)
    {
        var graphOutputs = new HashSet<string>(g.Outputs.Select(o => o.Name), StringComparer.Ordinal);
        var alias = new Dictionary<string, string>(StringComparer.Ordinal);
        var kept = new List<OnnxNode>(g.Nodes.Count);

        foreach (OnnxNode node in g.Nodes)
        {
            // A graph-output name is the caller's contract — keep its producer.
            bool elidable = IsPassthrough(node, g.OpsetVersion) && !graphOutputs.Contains(node.Outputs[0]);
            if (!elidable)
            {
                kept.Add(node);
                continue;
            }

            string source = node.Inputs[0];
            alias[node.Outputs[0]] = alias.TryGetValue(source, out string? deeper) ? deeper : source;
        }

        foreach (OnnxNode node in kept)
            for (int i = 0; i < node.Inputs.Length; i++)
                if (alias.TryGetValue(node.Inputs[i], out string? source))
                    node.Inputs[i] = source;

        g.Nodes = kept;
    }

    /// <summary>Identity, or Dropout in inference form (single used output, no
    /// training_mode input) — both forward input 0 unchanged.</summary>
    private static bool IsPassthrough(OnnxNode node, long opset)
    {
        if (node.Inputs.Length == 0 || node.Inputs[0].Length == 0 ||
            node.Outputs.Length == 0 || node.Outputs[0].Length == 0)
            return false;
        return node.OpType switch
        {
            "Identity" => true,
            // Dropout ≤ 6 signals inference via the is_test attribute (default 0
            // = training, where the op masks and scales — NOT a passthrough).
            "Dropout" => node.Inputs.Length <= 2 &&
                         node.Outputs.Count(o => o.Length > 0) == 1 &&
                         (opset >= 7 || node.GetInt("is_test") != 0),
            _ => false,
        };
    }

    // ── Dead-code elimination ───────────────────────────────────────────────────

    private static void RemoveDeadCode(OnnxGraph g)
    {
        var needed = new HashSet<string>(g.Outputs.Select(o => o.Name), StringComparer.Ordinal);

        var keptReversed = new List<OnnxNode>(g.Nodes.Count);
        for (int i = g.Nodes.Count - 1; i >= 0; i--)
        {
            OnnxNode node = g.Nodes[i];
            if (!node.Outputs.Any(o => o.Length > 0 && needed.Contains(o)))
                continue;
            keptReversed.Add(node);
            foreach (string input in node.Inputs)
                if (input.Length > 0)
                    needed.Add(input);
        }
        keptReversed.Reverse();
        g.Nodes = keptReversed;

        var initializers = new Dictionary<string, OnnxTensorData>(StringComparer.Ordinal);
        foreach ((string name, OnnxTensorData tensor) in g.Initializers)
            if (needed.Contains(name))
                initializers[name] = tensor;
        g.Initializers = initializers;
    }
}
