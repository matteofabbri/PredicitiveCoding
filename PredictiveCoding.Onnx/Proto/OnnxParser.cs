using System.Buffers.Binary;

namespace PredictiveCoding.Onnx.Proto;

// ─────────────────────────────────────────────────────────────────────────────
// OnnxParser — turns the raw bytes of a .onnx file into an OnnxGraph.
//
// A .onnx file is a protobuf-encoded ModelProto; this parser walks it with
// ProtoReader, keeping only the slice the executor needs: graph nodes with
// attributes, initializers (weights), declared inputs/outputs and the ai.onnx
// opset version. Field numbers below come straight from onnx.proto. Two
// protobuf realities are handled everywhere: repeated numeric fields may be
// PACKED (one length-delimited block) or UNPACKED (one value per tag), and
// unknown fields at any level are skipped cleanly, so files from newer
// exporters still load.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Parses ONNX model bytes (a protobuf ModelProto) into <see cref="OnnxGraph"/>.</summary>
public static class OnnxParser
{
    /// <summary>Parse the bytes of a .onnx file. Throws <see cref="InvalidDataException"/>
    /// when the input is not a well-formed ModelProto.</summary>
    public static OnnxGraph Parse(byte[] bytes)
    {
        // ModelProto: ir_version=1 (skipped), graph=7, opset_import=8.
        var reader = new ProtoReader(bytes);
        OnnxGraph? graph = null;
        long opsetVersion = 0;

        while (!reader.AtEnd)
        {
            var (field, wire) = reader.ReadTag();
            switch (field)
            {
                case 7 when wire == ProtoReader.WireLengthDelimited:
                    graph = ParseGraph(reader.ReadSubReader());
                    break;
                case 8 when wire == ProtoReader.WireLengthDelimited:
                    var (domain, version) = ParseOpsetImport(reader.ReadSubReader());
                    if (domain is "" or "ai.onnx") opsetVersion = version;
                    break;
                default:
                    reader.SkipField(wire);
                    break;
            }
        }

        if (graph is null)
            throw new InvalidDataException("No GraphProto found — the bytes are not a valid .onnx model.");
        graph.OpsetVersion = opsetVersion;
        return graph;
    }

    // OperatorSetIdProto: domain=1 (string), version=2 (varint int64).
    private static (string Domain, long Version) ParseOpsetImport(ProtoReader r)
    {
        string domain = "";
        long version = 0;
        while (!r.AtEnd)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1 when wire == ProtoReader.WireLengthDelimited: domain = r.ReadString(); break;
                case 2 when wire == ProtoReader.WireVarint: version = unchecked((long)r.ReadVarint()); break;
                default: r.SkipField(wire); break;
            }
        }
        return (domain, version);
    }

    // GraphProto: node=1, name=2, initializer=5, input=11, output=12 (value_info=13 and the rest: skipped).
    private static OnnxGraph ParseGraph(ProtoReader r)
    {
        var graph = new OnnxGraph();
        var declaredInputs = new List<OnnxValueInfo>();

        while (!r.AtEnd)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1 when wire == ProtoReader.WireLengthDelimited:
                    graph.Nodes.Add(ParseNode(r.ReadSubReader()));
                    break;
                case 2 when wire == ProtoReader.WireLengthDelimited:
                    graph.Name = r.ReadString();
                    break;
                case 5 when wire == ProtoReader.WireLengthDelimited:
                    var init = ParseTensor(r.ReadSubReader());
                    graph.Initializers[init.Name] = init;
                    break;
                case 11 when wire == ProtoReader.WireLengthDelimited:
                    declaredInputs.Add(ParseValueInfo(r.ReadSubReader()));
                    break;
                case 12 when wire == ProtoReader.WireLengthDelimited:
                    graph.Outputs.Add(ParseValueInfo(r.ReadSubReader()));
                    break;
                default:
                    r.SkipField(wire);
                    break;
            }
        }

        // Older exporters list initializers also as graph inputs; the caller only
        // feeds the true inputs, so drop any declared input backed by a weight.
        // (Filtered after the loop — protobuf fields may arrive in any order.)
        graph.Inputs = declaredInputs.Where(i => !graph.Initializers.ContainsKey(i.Name)).ToList();
        return graph;
    }

    // NodeProto: input=1, output=2, name=3, op_type=4, attribute=5 (domain=7 and the rest: skipped).
    private static OnnxNode ParseNode(ProtoReader r)
    {
        var node = new OnnxNode();
        var inputs = new List<string>();
        var outputs = new List<string>();

        while (!r.AtEnd)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1 when wire == ProtoReader.WireLengthDelimited: inputs.Add(r.ReadString()); break;
                case 2 when wire == ProtoReader.WireLengthDelimited: outputs.Add(r.ReadString()); break;
                case 3 when wire == ProtoReader.WireLengthDelimited: node.Name = r.ReadString(); break;
                case 4 when wire == ProtoReader.WireLengthDelimited: node.OpType = r.ReadString(); break;
                case 5 when wire == ProtoReader.WireLengthDelimited:
                    var attr = ParseAttribute(r.ReadSubReader());
                    node.Attributes[attr.Name] = attr;
                    break;
                default: r.SkipField(wire); break;
            }
        }

        node.Inputs = inputs.ToArray();
        node.Outputs = outputs.ToArray();
        return node;
    }

    // AttributeProto: name=1, f=2 (fixed32), i=3 (varint), s=4 (bytes→UTF-8), t=5 (TensorProto),
    // floats=7, ints=8, strings=9, type=20 (varint → OnnxAttributeType).
    private static OnnxAttribute ParseAttribute(ProtoReader r)
    {
        var attr = new OnnxAttribute();
        List<float>? floats = null;
        List<long>? ints = null;
        List<string>? strings = null;

        while (!r.AtEnd)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1 when wire == ProtoReader.WireLengthDelimited: attr.Name = r.ReadString(); break;
                case 2 when wire == ProtoReader.WireFixed32: attr.F = r.ReadFloat(); break;
                case 3 when wire == ProtoReader.WireVarint: attr.I = unchecked((long)r.ReadVarint()); break;
                case 4 when wire == ProtoReader.WireLengthDelimited: attr.S = r.ReadString(); break;
                case 5 when wire == ProtoReader.WireLengthDelimited: attr.T = ParseTensor(r.ReadSubReader()); break;
                case 7: ReadRepeatedFloat(ref r, wire, floats ??= []); break;
                case 8: ReadRepeatedVarint(ref r, wire, ints ??= []); break;
                case 9 when wire == ProtoReader.WireLengthDelimited: (strings ??= []).Add(r.ReadString()); break;
                case 20 when wire == ProtoReader.WireVarint: attr.Type = (OnnxAttributeType)r.ReadVarint(); break;
                default: r.SkipField(wire); break;
            }
        }

        if (floats is not null) attr.Floats = floats.ToArray();
        if (ints is not null) attr.Ints = ints.ToArray();
        if (strings is not null) attr.Strings = strings.ToArray();
        return attr;
    }

    // TensorProto: dims=1, data_type=2, float_data=4, int32_data=5, int64_data=7, name=8, raw_data=9.
    private static OnnxTensorData ParseTensor(ProtoReader r)
    {
        var tensor = new OnnxTensorData();
        var dims = new List<long>();
        List<float>? floatData = null;
        List<long>? int32Data = null;      // varints widen to long on read
        List<long>? int64Data = null;
        byte[]? rawData = null;

        while (!r.AtEnd)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1: ReadRepeatedVarint(ref r, wire, dims); break;
                case 2 when wire == ProtoReader.WireVarint: tensor.DataType = (int)r.ReadVarint(); break;
                case 4: ReadRepeatedFloat(ref r, wire, floatData ??= []); break;
                case 5: ReadRepeatedVarint(ref r, wire, int32Data ??= []); break;
                case 7: ReadRepeatedVarint(ref r, wire, int64Data ??= []); break;
                case 8 when wire == ProtoReader.WireLengthDelimited: tensor.Name = r.ReadString(); break;
                case 9 when wire == ProtoReader.WireLengthDelimited: rawData = r.ReadBytes(); break;
                default: r.SkipField(wire); break;
            }
        }

        tensor.Dims = dims.ToArray();

        // Materialise the payload: raw_data (little-endian) wins when non-empty,
        // otherwise the typed arrays. Other data types stay null — the executor
        // throws lazily only if such a tensor is actually used.
        switch (tensor.DataType)
        {
            case OnnxDataType.Float:
                tensor.Floats = rawData is { Length: > 0 }
                    ? RawToFloats(rawData, tensor.Name)
                    : floatData?.ToArray() ?? [];
                break;
            case OnnxDataType.Int64:
                tensor.Longs = rawData is { Length: > 0 }
                    ? RawToLongs(rawData, tensor.Name)
                    : int64Data?.ToArray() ?? [];
                break;
            case OnnxDataType.Int32:
                tensor.Longs = rawData is { Length: > 0 }
                    ? RawInt32ToLongs(rawData, tensor.Name)
                    : int32Data?.ToArray() ?? [];
                break;
        }
        return tensor;
    }

    // ValueInfoProto: name=1, type=2 (TypeProto).
    private static OnnxValueInfo ParseValueInfo(ProtoReader r)
    {
        var info = new OnnxValueInfo();
        while (!r.AtEnd)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1 when wire == ProtoReader.WireLengthDelimited: info.Name = r.ReadString(); break;
                case 2 when wire == ProtoReader.WireLengthDelimited: info.Shape = ParseTypeProto(r.ReadSubReader()); break;
                default: r.SkipField(wire); break;
            }
        }
        return info;
    }

    // TypeProto: tensor_type=1 (TypeProto.Tensor). Non-tensor types: no shape.
    private static int[] ParseTypeProto(ProtoReader r)
    {
        int[] shape = [];
        while (!r.AtEnd)
        {
            var (field, wire) = r.ReadTag();
            if (field == 1 && wire == ProtoReader.WireLengthDelimited) shape = ParseTensorType(r.ReadSubReader());
            else r.SkipField(wire);
        }
        return shape;
    }

    // TypeProto.Tensor: elem_type=1 (skipped), shape=2 (TensorShapeProto).
    private static int[] ParseTensorType(ProtoReader r)
    {
        int[] shape = [];
        while (!r.AtEnd)
        {
            var (field, wire) = r.ReadTag();
            if (field == 2 && wire == ProtoReader.WireLengthDelimited) shape = ParseTensorShape(r.ReadSubReader());
            else r.SkipField(wire);
        }
        return shape;
    }

    // TensorShapeProto: dim=1 (repeated Dimension).
    private static int[] ParseTensorShape(ProtoReader r)
    {
        var dims = new List<int>();
        while (!r.AtEnd)
        {
            var (field, wire) = r.ReadTag();
            if (field == 1 && wire == ProtoReader.WireLengthDelimited) dims.Add(ParseDimension(r.ReadSubReader()));
            else r.SkipField(wire);
        }
        return dims.ToArray();
    }

    // TensorShapeProto.Dimension: dim_value=1 (varint) or dim_param=2 (string).
    // A concrete dim_value wins; a symbolic/absent dimension becomes -1 (dynamic).
    private static int ParseDimension(ProtoReader r)
    {
        long value = -1;
        while (!r.AtEnd)
        {
            var (field, wire) = r.ReadTag();
            if (field == 1 && wire == ProtoReader.WireVarint) value = unchecked((long)r.ReadVarint());
            else r.SkipField(wire);
        }
        return value is >= 0 and <= int.MaxValue ? (int)value : -1;
    }

    // ── Repeated numeric fields: PACKED (wire 2) or UNPACKED (one value per tag) ──

    private static void ReadRepeatedFloat(ref ProtoReader r, int wireType, List<float> into)
    {
        switch (wireType)
        {
            case ProtoReader.WireFixed32:
                into.Add(r.ReadFloat());
                break;
            case ProtoReader.WireLengthDelimited:
                var packed = r.ReadSubReader();
                while (!packed.AtEnd) into.Add(packed.ReadFloat());
                break;
            default:
                throw new InvalidDataException($"Unexpected wire type {wireType} for a repeated float field at offset {r.Position}.");
        }
    }

    private static void ReadRepeatedVarint(ref ProtoReader r, int wireType, List<long> into)
    {
        switch (wireType)
        {
            case ProtoReader.WireVarint:
                into.Add(unchecked((long)r.ReadVarint()));
                break;
            case ProtoReader.WireLengthDelimited:
                var packed = r.ReadSubReader();
                while (!packed.AtEnd) into.Add(unchecked((long)packed.ReadVarint()));
                break;
            default:
                throw new InvalidDataException($"Unexpected wire type {wireType} for a repeated integer field at offset {r.Position}.");
        }
    }

    // ── raw_data decoding: little-endian, fixed element width ──

    private static float[] RawToFloats(byte[] raw, string name)
    {
        if (raw.Length % 4 != 0)
            throw new InvalidDataException($"Tensor '{name}': raw_data length {raw.Length} is not a multiple of 4 (Float).");
        var values = new float[raw.Length / 4];
        for (int i = 0; i < values.Length; i++)
            values[i] = BinaryPrimitives.ReadSingleLittleEndian(raw.AsSpan(i * 4, 4));
        return values;
    }

    private static long[] RawToLongs(byte[] raw, string name)
    {
        if (raw.Length % 8 != 0)
            throw new InvalidDataException($"Tensor '{name}': raw_data length {raw.Length} is not a multiple of 8 (Int64).");
        var values = new long[raw.Length / 8];
        for (int i = 0; i < values.Length; i++)
            values[i] = BinaryPrimitives.ReadInt64LittleEndian(raw.AsSpan(i * 8, 8));
        return values;
    }

    private static long[] RawInt32ToLongs(byte[] raw, string name)
    {
        if (raw.Length % 4 != 0)
            throw new InvalidDataException($"Tensor '{name}': raw_data length {raw.Length} is not a multiple of 4 (Int32).");
        var values = new long[raw.Length / 4];
        for (int i = 0; i < values.Length; i++)
            values[i] = BinaryPrimitives.ReadInt32LittleEndian(raw.AsSpan(i * 4, 4));
        return values;
    }
}
