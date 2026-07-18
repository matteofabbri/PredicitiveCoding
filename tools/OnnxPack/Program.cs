using PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// OnnxPack — converts a .onnx model into the in-house PCNX format.
//
//   OnnxPack <input.onnx> [output.pcnx] [--no-optimize]
//
// With optimization (the default) the graph goes through GraphOptimizer:
// constant folding via the runtime's own kernels, Identity elision and
// dead-code removal — Run results stay bit-identical. The produced file loads
// through the same OnnxModel.Load (the format is sniffed from the content).
// ─────────────────────────────────────────────────────────────────────────────

bool optimize = !args.Contains("--no-optimize");
string[] unknownFlags = args.Where(a => a.StartsWith("--") && a != "--no-optimize").ToArray();
string[] paths = args.Where(a => !a.StartsWith("--")).ToArray();

if (unknownFlags.Length > 0)
{
    Console.Error.WriteLine($"Unknown option(s): {string.Join(", ", unknownFlags)}. Did you mean --no-optimize?");
    return 1;
}

if (paths.Length is < 1 or > 2)
{
    Console.WriteLine("Convert a .onnx model to the in-house PCNX format (no protobuf at load).");
    Console.WriteLine();
    Console.WriteLine("  Usage: OnnxPack <input.onnx> [output.pcnx] [--no-optimize]");
    Console.WriteLine();
    Console.WriteLine("  Default output: the input path with the .pcnx extension.");
    return 1;
}

string input = paths[0];
string output = paths.Length == 2 ? paths[1] : Path.ChangeExtension(input, PcnxFormat.Extension);

if (!File.Exists(input))
{
    Console.Error.WriteLine($"Input not found: {input}");
    return 1;
}

if (string.Equals(Path.GetFullPath(input), Path.GetFullPath(output), StringComparison.OrdinalIgnoreCase))
{
    Console.Error.WriteLine("Output path equals the input path; pick a different output file.");
    return 1;
}

OnnxModel model = OnnxModel.Load(input);
int nodesBefore = model.Graph.Nodes.Count;
int initsBefore = model.Graph.Initializers.Count;

OnnxModel packed = optimize ? model.Optimized() : model;
packed.Save(output, optimize: false);   // already optimized (or explicitly not)

// Reload what we just wrote: proves the file parses and gives honest stats.
OnnxModel reloaded = OnnxModel.Load(output);

Console.WriteLine($"  {input}  →  {output}");
Console.WriteLine($"  nodes        : {nodesBefore} → {reloaded.Graph.Nodes.Count}" + (optimize ? "" : "  (no optimization)"));
Console.WriteLine($"  initializers : {initsBefore} → {reloaded.Graph.Initializers.Count}");
Console.WriteLine($"  file size    : {new FileInfo(input).Length / 1_048_576.0:F1} MB → {new FileInfo(output).Length / 1_048_576.0:F1} MB");
Console.WriteLine($"  inputs       : {string.Join(", ", reloaded.InputNames)}");
Console.WriteLine($"  outputs      : {string.Join(", ", reloaded.OutputNames)}");
return 0;
