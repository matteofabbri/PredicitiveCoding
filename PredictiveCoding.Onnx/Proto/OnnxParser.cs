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
    /// when the input is not a well-formed ModelProto.
    /// <paramref name="externalDataDir"/> is the directory used to resolve tensors stored
    /// in the "external data" format (data_location=EXTERNAL + a side file like
    /// <c>model.onnx_data</c>); null means such tensors fail with a clear message.</summary>
    public static OnnxGraph Parse(byte[] bytes, string? externalDataDir = null)
    {
        // ModelProto: ir_version=1 (skipped), graph=7, opset_import=8.
        var reader = new ProtoReader(bytes);
        OnnxGraph? graph = null;
        long opsetVersion = 0;
        using var external = new ExternalDataSource(externalDataDir);

        while (!reader.AtEnd)
        {
            var (field, wire) = reader.ReadTag();
            switch (field)
            {
                case 7 when wire == ProtoReader.WireLengthDelimited:
                    graph = ParseGraph(reader.ReadSubReader(), external);
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
    private static OnnxGraph ParseGraph(ProtoReader r, ExternalDataSource external)
    {
        var graph = new OnnxGraph();
        var declaredInputs = new List<OnnxValueInfo>();

        while (!r.AtEnd)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1 when wire == ProtoReader.WireLengthDelimited:
                    graph.Nodes.Add(ParseNode(r.ReadSubReader(), external));
                    break;
                case 2 when wire == ProtoReader.WireLengthDelimited:
                    graph.Name = r.ReadString();
                    break;
                case 5 when wire == ProtoReader.WireLengthDelimited:
                    var init = ParseTensor(r.ReadSubReader(), external);
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
    private static OnnxNode ParseNode(ProtoReader r, ExternalDataSource external)
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
                    var attr = ParseAttribute(r.ReadSubReader(), external);
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
    private static OnnxAttribute ParseAttribute(ProtoReader r, ExternalDataSource external)
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
                case 5 when wire == ProtoReader.WireLengthDelimited: attr.T = ParseTensor(r.ReadSubReader(), external); break;
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

    // TensorProto: dims=1, data_type=2, float_data=4, int32_data=5, int64_data=7, name=8,
    // raw_data=9, external_data=13 (repeated StringStringEntryProto), data_location=14.
    private static OnnxTensorData ParseTensor(ProtoReader r, ExternalDataSource external)
    {
        var tensor = new OnnxTensorData();
        var dims = new List<long>();
        List<float>? floatData = null;
        List<long>? int32Data = null;      // varints widen to long on read
        List<long>? int64Data = null;
        byte[]? rawData = null;
        long dataLocation = 0;             // 0=DEFAULT (inline), 1=EXTERNAL
        string? extLocation = null;        // side-file path, relative to the model
        long extOffset = 0, extLength = -1;

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
                case 13 when wire == ProtoReader.WireLengthDelimited:
                    var (key, value) = ParseStringEntry(r.ReadSubReader());
                    switch (key)
                    {
                        case "location": extLocation = value; break;
                        case "offset": extOffset = long.Parse(value); break;
                        case "length": extLength = long.Parse(value); break;
                        // "checksum" and unknown keys: ignored
                    }
                    break;
                case 14 when wire == ProtoReader.WireVarint: dataLocation = unchecked((long)r.ReadVarint()); break;
                default: r.SkipField(wire); break;
            }
        }

        tensor.Dims = dims.ToArray();

        // External storage (data_location=EXTERNAL): the payload lives in a side file
        // (e.g. model.onnx_data) at offset/length; read it straight into the final array.
        if (dataLocation == 1)
        {
            if (extLocation is null)
                throw new InvalidDataException($"Tensor '{tensor.Name}': data_location is EXTERNAL but no 'location' entry is present.");
            external.Fill(tensor, extLocation, extOffset, extLength);
            return tensor;
        }

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

    // StringStringEntryProto: key=1, value=2.
    private static (string Key, string Value) ParseStringEntry(ProtoReader r)
    {
        string key = "", value = "";
        while (!r.AtEnd)
        {
            var (field, wire) = r.ReadTag();
            switch (field)
            {
                case 1 when wire == ProtoReader.WireLengthDelimited: key = r.ReadString(); break;
                case 2 when wire == ProtoReader.WireLengthDelimited: value = r.ReadString(); break;
                default: r.SkipField(wire); break;
            }
        }
        return (key, value);
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

    // ─────────────────────────────────────────────────────────────────────────
    // External data: tensors whose payload lives in a side file next to the
    // model (data_location=EXTERNAL). Streams are opened once per side file and
    // shared across the (potentially hundreds of) tensors that point into it;
    // payloads decode straight into the destination array — with multi-GB
    // weight files an intermediate byte[] copy would double peak memory.
    // ─────────────────────────────────────────────────────────────────────────
    private sealed class ExternalDataSource(string? baseDir) : IDisposable
    {
        private readonly Dictionary<string, FileStream> _files = new(StringComparer.OrdinalIgnoreCase);

        public void Fill(OnnxTensorData tensor, string location, long offset, long length)
        {
            if (baseDir is null)
                throw new InvalidDataException(
                    $"Tensor '{tensor.Name}' uses external data ('{location}') but the model was parsed from bytes " +
                    "with no directory context. Load it via OnnxModel.Load(path) so the side file can be resolved.");
            if (!BitConverter.IsLittleEndian)
                throw new PlatformNotSupportedException("External tensor data assumes a little-endian host.");

            long count = tensor.Dims.Aggregate(1L, (a, d) => a * d);
            switch (tensor.DataType)
            {
                case OnnxDataType.Float:
                    var floats = new float[count];
                    ReadInto(location, offset, length, System.Runtime.InteropServices.MemoryMarshal.AsBytes(floats.AsSpan()), tensor.Name);
                    tensor.Floats = floats;
                    break;
                case OnnxDataType.Int64:
                    var longs = new long[count];
                    ReadInto(location, offset, length, System.Runtime.InteropServices.MemoryMarshal.AsBytes(longs.AsSpan()), tensor.Name);
                    tensor.Longs = longs;
                    break;
                case OnnxDataType.Int32:
                    var raw = new byte[checked(count * 4)];
                    ReadInto(location, offset, length, raw, tensor.Name);
                    tensor.Longs = RawInt32ToLongs(raw, tensor.Name);
                    break;
                // Other dtypes: leave the payload null — the executor throws only on use.
            }
        }

        private void ReadInto(string location, long offset, long length, Span<byte> destination, string tensorName)
        {
            if (length >= 0 && length != destination.Length)
                throw new InvalidDataException(
                    $"Tensor '{tensorName}': external data length {length} does not match the declared dims ({destination.Length} bytes).");

            if (!_files.TryGetValue(location, out FileStream? fs))
            {
                string path = Path.Combine(baseDir!, location);
                if (!File.Exists(path))
                    throw new InvalidDataException($"Tensor '{tensorName}': external data file '{path}' not found.");
                fs = File.OpenRead(path);
                _files[location] = fs;
            }

            fs.Seek(offset, SeekOrigin.Begin);
            fs.ReadExactly(destination);
        }

        public void Dispose()
        {
            foreach (FileStream fs in _files.Values) fs.Dispose();
        }
    }
}
