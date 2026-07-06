using PredictiveCoding.Onnx.Proto;

namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// Executor — walks the graph node by node and dispatches to the Ops kernels.
//
// ONNX requires nodes to be topologically sorted in the file, so a single
// forward pass suffices; we still verify every input is resolved before use so
// a malformed file fails with a clear message instead of a null tensor.
// ─────────────────────────────────────────────────────────────────────────────

public static class Executor
{
    public static Dictionary<string, Tensor> Run(OnnxGraph graph, Dictionary<string, Tensor> inputs)
    {
        // Value environment: initializers (weights) + caller inputs, then node outputs as they compute.
        var env = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var (name, init) in graph.Initializers) env[name] = init.ToTensor();
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
                if (!env.TryGetValue(name, out args[i]!))
                    throw new InvalidOperationException($"Node '{node.Name}' ({node.OpType}) needs value '{name}' which no earlier node produced.");
            }

            Tensor[] results = Dispatch(node, args);

            if (results.Length < node.Outputs.Length)
                throw new InvalidOperationException($"Op {node.OpType} returned {results.Length} outputs, node declares {node.Outputs.Length}.");
            for (int i = 0; i < node.Outputs.Length; i++)
                if (node.Outputs[i].Length > 0)
                    env[node.Outputs[i]] = results[i];
        }

        var outputs = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var o in graph.Outputs)
        {
            if (!env.TryGetValue(o.Name, out var t))
                throw new InvalidOperationException($"Graph output '{o.Name}' was never produced.");
            outputs[o.Name] = t;
        }
        return outputs;
    }

    private static Tensor[] Dispatch(OnnxNode n, Tensor[] a) => n.OpType switch
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
        "Softmax"              => [Ops.Softmax(a[0], (int)n.GetInt("axis", -1))],

        // Convolution / pooling
        "Conv"                 => [Ops.Conv(a[0], a[1], a.Length > 2 ? a[2] : null,
                                            n.GetInts("strides"), n.GetInts("pads"),
                                            n.GetInts("dilations"), n.GetInt("group", 1))],
        "MaxPool"              => [Ops.MaxPool(a[0], n.GetInts("kernel_shape"),
                                               n.GetInts("strides"), n.GetInts("pads"),
                                               n.GetInt("ceil_mode", 0) != 0)],
        "AveragePool"          => [Ops.AveragePool(a[0], n.GetInts("kernel_shape"),
                                                   n.GetInts("strides"), n.GetInts("pads"),
                                                   n.GetInt("count_include_pad", 0) != 0)],
        "GlobalAveragePool"    => [Ops.GlobalAveragePool(a[0])],

        // Linear algebra / normalisation
        "Gemm"                 => [Ops.Gemm(a[0], a[1], a.Length > 2 ? a[2] : null,
                                            n.GetFloat("alpha", 1f), n.GetFloat("beta", 1f),
                                            n.GetInt("transA", 0) != 0, n.GetInt("transB", 0) != 0)],
        "MatMul"               => [Ops.MatMul(a[0], a[1])],
        "BatchNormalization"   => [Ops.BatchNorm(a[0], a[1], a[2], a[3], a[4],
                                                 n.GetFloat("epsilon", 1e-5f))],

        // Shape plumbing
        "Reshape"              => [Ops.Reshape(a[0], a[1])],
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
        "Unsqueeze"            => [Ops.Unsqueeze(a[0], a.Length > 1 ? a[1] : null, n.GetInts("axes"))],
        "Squeeze"              => [Ops.Squeeze(a[0], a.Length > 1 ? a[1] : null, n.GetInts("axes"))],
        "Identity"             => [a[0]],

        _ => throw new NotSupportedException(
                 $"Op '{n.OpType}' (node '{n.Name}') is not implemented by the in-house ONNX executor."),
    };
}
