using PredictiveCoding.Onnx.Proto;

namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// OnnxModel — the public face of the in-house ONNX runtime.
//
//   var model  = OnnxModel.Load("yolox.onnx");
//   var output = model.Run(new() { ["images"] = inputTensor })["output"];
//   model.Save("yolox.pcnx");                 // optimized in-house format
//   var again  = OnnxModel.Load("yolox.pcnx"); // Load sniffs the format
//
// No ONNX Runtime, no protobuf package: OnnxParser reads .onnx files, PcnxFormat
// reads/writes the in-house PCNX format, Executor walks the graph calling the
// Ops kernels. Supported op set is intentionally small (see Executor) — enough
// for the models this repo ships (YOLOX first, ArcFace/RetinaFace-class nets
// next); unsupported ops fail loudly at Run.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class OnnxModel
{
    public OnnxGraph Graph { get; }

    private OnnxModel(OnnxGraph graph) => Graph = graph;

    /// <summary>Loads a model from disk — either a .onnx protobuf or a PCNX file
    /// (the format is detected from the content, not the extension). The model's own
    /// directory resolves any external-data side files (e.g. <c>model.onnx_data</c>).
    /// PCNX reads stream from disk, so files may exceed 2 GB; a .onnx protobuf cannot
    /// (protobuf's own limit) — its weights land in a side file instead.</summary>
    public static OnnxModel Load(string path)
    {
        using FileStream fs = File.OpenRead(path);
        Span<byte> magic = stackalloc byte[4];
        int got = fs.ReadAtLeast(magic, magic.Length, throwOnEndOfStream: false);
        fs.Seek(0, SeekOrigin.Begin);
        if (got == magic.Length && PcnxFormat.LooksLikePcnx(magic))
            return new(PcnxFormat.Read(fs));

        if (fs.Length > int.MaxValue)
            throw new InvalidDataException(
                $"'{path}' is a {fs.Length / 1_048_576} MB .onnx protobuf — beyond the 2 GB protobuf limit, so it cannot be a valid model. (PCNX files this large are fine.)");
        byte[] bytes = new byte[fs.Length];
        fs.ReadExactly(bytes);
        return new(OnnxParser.Parse(bytes, Path.GetDirectoryName(Path.GetFullPath(path))));
    }

    /// <inheritdoc cref="Load(string)"/> External-data tensors cannot be resolved from
    /// bare bytes — use <see cref="Load(string)"/> for models with a side file.
    public static OnnxModel Load(byte[] bytes)
        => new(PcnxFormat.LooksLikePcnx(bytes) ? PcnxFormat.Read(bytes) : OnnxParser.Parse(bytes));

    /// <summary>
    /// Saves the model in the in-house PCNX format. With <paramref name="optimize"/>
    /// (the default) the graph is first put through <see cref="GraphOptimizer"/> —
    /// constant folding via the runtime's own kernels, Identity elision and
    /// dead-code removal — all of which leave <see cref="Run"/> results bit-identical.
    /// </summary>
    public void Save(string path, bool optimize = true)
        => PcnxFormat.Write(path, optimize ? GraphOptimizer.Optimize(Graph) : Graph);

    /// <summary>A copy of this model with the save-time optimizations applied in memory.</summary>
    public OnnxModel Optimized() => new(GraphOptimizer.Optimize(Graph));

    /// <summary>Names the caller must feed (graph inputs minus initializers).</summary>
    public IReadOnlyList<string> InputNames => Graph.Inputs.Select(i => i.Name).ToArray();

    public IReadOnlyList<string> OutputNames => Graph.Outputs.Select(o => o.Name).ToArray();

    /// <summary>Run the graph. Returns every declared graph output by name.</summary>
    public Dictionary<string, Tensor> Run(Dictionary<string, Tensor> inputs)
        => Executor.Run(Graph, inputs);
}
