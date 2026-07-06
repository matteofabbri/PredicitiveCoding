using PredictiveCoding.Onnx.Proto;

namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// OnnxModel — the public face of the in-house ONNX runtime.
//
//   var model  = OnnxModel.Load("yolox.onnx");
//   var output = model.Run(new() { ["images"] = inputTensor })["output"];
//
// No ONNX Runtime, no protobuf package: OnnxParser reads the file, Executor
// walks the graph calling the Ops kernels. Supported op set is intentionally
// small (see Executor) — enough for the models this repo ships (YOLOX first,
// ArcFace/RetinaFace-class nets next); unsupported ops fail loudly at Run.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class OnnxModel
{
    public OnnxGraph Graph { get; }

    private OnnxModel(OnnxGraph graph) => Graph = graph;

    public static OnnxModel Load(string path) => new(OnnxParser.Parse(File.ReadAllBytes(path)));

    public static OnnxModel Load(byte[] bytes) => new(OnnxParser.Parse(bytes));

    /// <summary>Names the caller must feed (graph inputs minus initializers).</summary>
    public IReadOnlyList<string> InputNames => Graph.Inputs.Select(i => i.Name).ToArray();

    public IReadOnlyList<string> OutputNames => Graph.Outputs.Select(o => o.Name).ToArray();

    /// <summary>Run the graph. Returns every declared graph output by name.</summary>
    public Dictionary<string, Tensor> Run(Dictionary<string, Tensor> inputs)
        => Executor.Run(Graph, inputs);
}
