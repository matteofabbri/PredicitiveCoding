namespace PredictiveCoding.Onnx.Proto;

// ─────────────────────────────────────────────────────────────────────────────
// In-memory picture of an ONNX file — the parser's output, the executor's input.
//
// This mirrors just the slice of onnx.proto we need: the graph, its nodes with
// attributes, the initializers (weights) and the declared inputs/outputs.
// Field numbers live in OnnxParser; these types are protobuf-agnostic.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>A parsed .onnx file: the graph plus the model-level opset version.</summary>
public sealed class OnnxGraph
{
    public string Name = "";
    public long OpsetVersion;                       // ai.onnx domain opset (0 if absent)

    public List<OnnxNode> Nodes = [];
    public Dictionary<string, OnnxTensorData> Initializers = new(StringComparer.Ordinal);

    /// <summary>Graph inputs that are NOT initializers — i.e. what the caller must feed.</summary>
    public List<OnnxValueInfo> Inputs = [];
    public List<OnnxValueInfo> Outputs = [];
}

/// <summary>One operation in the graph (a NodeProto).</summary>
public sealed class OnnxNode
{
    public string Name = "";
    public string OpType = "";
    public string[] Inputs = [];                    // value names ("" = optional input omitted)
    public string[] Outputs = [];
    public Dictionary<string, OnnxAttribute> Attributes = new(StringComparer.Ordinal);

    /// <summary>Int attribute with default — the accessor virtually every op needs.</summary>
    public long GetInt(string name, long fallback = 0)
        => Attributes.TryGetValue(name, out var a) ? a.I : fallback;

    public long[] GetInts(string name, long[]? fallback = null)
        => Attributes.TryGetValue(name, out var a) && a.Ints is not null ? a.Ints : fallback ?? [];

    public float GetFloat(string name, float fallback = 0f)
        => Attributes.TryGetValue(name, out var a) ? a.F : fallback;

    public string GetString(string name, string fallback = "")
        => Attributes.TryGetValue(name, out var a) && a.S is not null ? a.S : fallback;
}

/// <summary>A node attribute (AttributeProto) — only the payload slot matching Type is set.</summary>
public sealed class OnnxAttribute
{
    public string Name = "";
    public OnnxAttributeType Type;
    public float F;
    public long I;
    public string? S;
    public float[]? Floats;
    public long[]? Ints;
    public string[]? Strings;
    public OnnxTensorData? T;
}

public enum OnnxAttributeType
{
    Undefined = 0, Float = 1, Int = 2, String = 3, Tensor = 4, Graph = 5,
    Floats = 6, Ints = 7, Strings = 8, Tensors = 9, Graphs = 10,
}

/// <summary>A constant tensor from the file (TensorProto): an initializer or a tensor attribute.</summary>
public sealed class OnnxTensorData
{
    public string Name = "";
    public int DataType;                            // OnnxDataType values
    public long[] Dims = [];
    public float[]? Floats;                         // set when DataType == Float
    public long[]? Longs;                           // set when DataType == Int64 / Int32 (widened)

    /// <summary>Convert to a runtime tensor. Rank-0 scalars are widened to shape [1].</summary>
    public Tensor ToTensor()
    {
        int[] shape = Dims.Length == 0 ? [1] : Array.ConvertAll(Dims, d => checked((int)d));
        return Floats is not null ? new Tensor(shape, Floats)
             : Longs is not null ? new Tensor(shape, Longs)
             : throw new NotSupportedException($"Initializer '{Name}' has unsupported data type {DataType}.");
    }
}

/// <summary>ONNX TensorProto.DataType values we care about.</summary>
public static class OnnxDataType
{
    public const int Float = 1;
    public const int Int32 = 6;
    public const int Int64 = 7;
}

/// <summary>A declared graph input/output: name + static shape when known (-1 for dynamic dims).</summary>
public sealed class OnnxValueInfo
{
    public string Name = "";
    public int[] Shape = [];
}
