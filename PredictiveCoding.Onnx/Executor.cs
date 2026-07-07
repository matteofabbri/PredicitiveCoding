using PredictiveCoding.Onnx.Proto;

namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// Executor — walks the graph node by node and dispatches to the Ops kernels.
//
// ONNX requires nodes to be topologically sorted in the file, so a single
// forward pass suffices; we still verify every input is resolved before use so
// a malformed file fails with a clear message instead of a null tensor.
//
// Initializers materialise lazily on first use: a model may carry weights of
// data types we don't support (fp16/double/int8 quantisation leftovers) on
// paths that never execute — those must not block the paths that do.
// ─────────────────────────────────────────────────────────────────────────────

public static class Executor
{
    public static Dictionary<string, Tensor> Run(OnnxGraph graph, Dictionary<string, Tensor> inputs)
    {
        // Value environment: caller inputs up front; initializers (weights) join
        // lazily via Resolve; node outputs are added as they compute.
        var env = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var (name, tensor) in inputs) env[name] = tensor;

        foreach (var missing in graph.Inputs.Where(i => !env.ContainsKey(i.Name)))
            throw new ArgumentException($"Missing graph input '{missing.Name}'. Expected inputs: {string.Join(", ", graph.Inputs.Select(i => i.Name))}.");

        foreach (var node in graph.Nodes)
        {
            Tensor[] args = new Tensor[node.Inputs.Length];
            for (int i = 0; i < node.Inputs.Length; i++)
            {
                string name = node.Inputs[i];
                if (name.Length == 0) continue;      // optional input omitted → slot stays null
                args[i] = Resolve(graph, env, name)
                    ?? throw new InvalidOperationException($"Node '{node.Name}' ({node.OpType}) needs value '{name}' which no earlier node produced.");
            }

            Tensor[] results = Dispatch(node, args, graph.OpsetVersion);

            if (results.Length < node.Outputs.Length)
                throw new InvalidOperationException($"Op {node.OpType} returned {results.Length} outputs, node declares {node.Outputs.Length}.");
            for (int i = 0; i < node.Outputs.Length; i++)
                if (node.Outputs[i].Length > 0)
                    env[node.Outputs[i]] = results[i];
        }

        var outputs = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var o in graph.Outputs)
            outputs[o.Name] = Resolve(graph, env, o.Name)
                ?? throw new InvalidOperationException($"Graph output '{o.Name}' was never produced.");
        return outputs;
    }

    /// <summary>Look a value up in the environment, falling back to lazily converting an
    /// initializer of that name (cached in env so each weight converts once). Null if unknown.</summary>
    private static Tensor? Resolve(OnnxGraph graph, Dictionary<string, Tensor> env, string name)
    {
        if (env.TryGetValue(name, out var t)) return t;
        if (!graph.Initializers.TryGetValue(name, out var init)) return null;
        var tensor = init.ToTensor();                // throws only for dtypes we can't represent
        env[name] = tensor;
        return tensor;
    }

    private static Tensor[] Dispatch(OnnxNode n, Tensor[] a, long opset) => n.OpType switch
    {
        // Elementwise / activation
        "Add"                  => [Ops.Add(a[0], a[1])],
        "Sub"                  => [Ops.Sub(a[0], a[1])],
        "Mul"                  => [Ops.Mul(a[0], a[1])],
        "Div"                  => [Ops.Div(a[0], a[1])],
        "Sigmoid"              => [Ops.Sigmoid(a[0])],
        "Relu"                 => [Ops.Relu(a[0])],
        "PRelu"                => [Ops.PRelu(a[0], a[1])],
        "Exp"                  => [Ops.Exp(a[0])],
        "Sqrt"                 => [Ops.Sqrt(a[0])],
        "Softmax"              => [Ops.Softmax(a[0], SoftmaxAxis(n, a[0], opset))],

        // Convolution / pooling. auto_pad SAME_UPPER/SAME_LOWER is resolved to explicit
        // pads here so the kernels only ever see [top, left, bottom, right].
        "Conv"                 => [Ops.Conv(a[0], a[1], a.Length > 2 ? a[2] : null,
                                            n.GetInts("strides"),
                                            ResolvePads(n, a[0], [a[1].Shape[2], a[1].Shape[3]], n.GetInts("dilations")),
                                            n.GetInts("dilations"), n.GetInt("group", 1))],
        "MaxPool"              => [Ops.MaxPool(a[0], PoolKernel(n),
                                               n.GetInts("strides"),
                                               ResolvePads(n, a[0], PoolKernel(n), dilations: []),
                                               n.GetInt("ceil_mode", 0) != 0)],
        "AveragePool"          => [Ops.AveragePool(a[0], PoolKernel(n),
                                                   n.GetInts("strides"),
                                                   ResolvePads(n, a[0], PoolKernel(n), dilations: []),
                                                   n.GetInt("count_include_pad", 0) != 0)],
        "GlobalAveragePool"    => [Ops.GlobalAveragePool(a[0])],

        // Resampling. Opset 10: Resize(X, scales); opset 11+: Resize(X, roi, scales[, sizes]).
        "Resize"               => [Ops.Resize(a[0],
                                              scales: a.Length == 2 ? a[1] : a.Length > 2 ? a[2] : null,
                                              sizes: a.Length > 3 ? a[3] : null,
                                              mode: n.GetString("mode", "nearest"),
                                              coordMode: n.GetString("coordinate_transformation_mode", "half_pixel"),
                                              nearestMode: n.GetString("nearest_mode", "round_prefer_floor"))],

        // Linear algebra / normalisation
        "Gemm"                 => [Ops.Gemm(a[0], a[1], a.Length > 2 ? a[2] : null,
                                            n.GetFloat("alpha", 1f), n.GetFloat("beta", 1f),
                                            n.GetInt("transA", 0) != 0, n.GetInt("transB", 0) != 0)],
        "MatMul"               => [Ops.MatMul(a[0], a[1])],
        "BatchNormalization"   => [Ops.BatchNorm(a[0], a[1], a[2], a[3], a[4],
                                                 n.GetFloat("epsilon", 1e-5f))],

        // Shape plumbing
        "Reshape"              => n.GetInt("allowzero", 0) == 0
                                  ? [Ops.Reshape(a[0], a[1])]
                                  : throw new NotSupportedException("Reshape with allowzero=1 is not supported."),
        "Flatten"              => [Ops.Flatten(a[0], (int)n.GetInt("axis", 1))],
        "Transpose"            => [Ops.Transpose(a[0], n.GetInts("perm"))],
        "Concat"               => [Ops.Concat(a, (int)n.GetInt("axis"))],
        "Slice"                => a.Length >= 3
                                  ? [Ops.Slice(a[0], a[1], a[2],
                                               a.Length > 3 ? a[3] : null,
                                               a.Length > 4 ? a[4] : null)]
                                  : throw new NotSupportedException(
                                        "Slice with attribute-form parameters (opset < 10) is not supported."),
        "Shape"                => [Ops.Shape(a[0])],
        "Gather"               => [Ops.Gather(a[0], a[1], (int)n.GetInt("axis", 0))],
        "Unsqueeze"            => [Ops.Unsqueeze(a[0], a.Length > 1 ? a[1] : null, n.GetInts("axes"))],
        "Squeeze"              => [Ops.Squeeze(a[0], a.Length > 1 ? a[1] : null, n.GetInts("axes"))],
        "Identity"             => [a[0]],

        _ => throw new NotSupportedException(
                 $"Op '{n.OpType}' (node '{n.Name}') is not implemented by the in-house ONNX executor."),
    };

    /// <summary>kernel_shape for pools, with the guards for attributes our kernels ignore.</summary>
    private static long[] PoolKernel(OnnxNode n)
    {
        long[] dil = n.GetInts("dilations");
        if (dil.Any(d => d != 1))
            throw new NotSupportedException($"{n.OpType} with dilations != 1 is not supported.");
        long[] kernel = n.GetInts("kernel_shape");
        if (kernel.Length == 0)
            throw new NotSupportedException($"{n.OpType} node '{n.Name}' is missing kernel_shape.");
        return kernel;
    }

    /// <summary>
    /// Effective [top, left, bottom, right] padding: explicit 'pads' for NOTSET, zeros for
    /// VALID, and the ONNX SAME_UPPER/SAME_LOWER formula otherwise — per axis,
    /// total = max(0, (ceil(in/stride)−1)·stride + (k−1)·dilation + 1 − in), split evenly
    /// with the odd pixel at the end (UPPER) or the start (LOWER).
    /// </summary>
    private static long[] ResolvePads(OnnxNode n, Tensor x, long[] kernel, long[] dilations)
    {
        string autoPad = n.GetString("auto_pad", "NOTSET");
        switch (autoPad)
        {
            case "NOTSET":
                return n.GetInts("pads");
            case "VALID":
                return [0, 0, 0, 0];
            case "SAME_UPPER":
            case "SAME_LOWER":
                if (x.Rank != 4)
                    throw new NotSupportedException($"{n.OpType} auto_pad '{autoPad}' supports 4-D NCHW input only, got {x.ShapeString}.");
                long[] strides = n.GetInts("strides");
                var pads = new long[4];
                for (int i = 0; i < 2; i++)
                {
                    long inDim = x.Shape[2 + i];
                    long stride = strides.Length > i ? strides[i] : 1;
                    long dilation = dilations.Length > i ? dilations[i] : 1;
                    long effKernel = (kernel[i] - 1) * dilation + 1;
                    long outDim = (inDim + stride - 1) / stride;             // ceil(in / stride)
                    long total = Math.Max(0, (outDim - 1) * stride + effKernel - inDim);
                    long lo = autoPad == "SAME_LOWER" ? (total + 1) / 2 : total / 2;
                    pads[i] = lo;                                            // top / left
                    pads[2 + i] = total - lo;                                // bottom / right
                }
                return pads;
            default:
                throw new NotSupportedException($"{n.OpType} auto_pad '{autoPad}' is not supported.");
        }
    }

    /// <summary>
    /// Softmax axis with opset semantics: opset ≥ 13 is a true single-axis softmax
    /// (default −1); before that, ONNX defined Softmax as flatten-at-axis (default 1),
    /// which only coincides with the single-axis kernel when axis is the last dimension —
    /// anything else fails loudly rather than computing subtly different numbers.
    /// </summary>
    private static int SoftmaxAxis(OnnxNode n, Tensor x, long opset)
    {
        int axis = (int)n.GetInt("axis", opset >= 13 ? -1 : 1);
        int normalized = axis < 0 ? axis + x.Rank : axis;
        if (opset < 13 && opset > 0 && normalized != x.Rank - 1)
            throw new NotSupportedException(
                $"Softmax with opset {opset} flatten semantics and axis {axis} (not the last dimension) is not supported.");
        return axis;
    }
}
