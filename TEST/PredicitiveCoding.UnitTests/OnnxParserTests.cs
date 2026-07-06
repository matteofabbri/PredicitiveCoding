using System.Text;
using PredictiveCoding.Onnx.Proto;

namespace PredictiveCoding.Onnx.Tests;

// ─────────────────────────────────────────────────────────────────────────────
// OnnxParserTests — byte-level tests for the in-house ONNX (protobuf) parser.
//
// No protobuf library and no .onnx file on disk: a tiny private wire-format
// writer (MiniProtoWriter) byte-builds real ModelProto payloads, so these tests
// pin down exactly what the parser must accept per the protobuf wire spec —
// packed and unpacked repeated fields, raw_data vs typed arrays, unknown-field
// skipping, dynamic dims and the initializer/graph-input bookkeeping.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Wire-format tests for <see cref="OnnxModel.Load(byte[])"/>: hand-built
/// ModelProto bytes go in, a parsed <see cref="OnnxGraph"/> must come out.</summary>
public class OnnxParserTests
{
    // Protobuf wire types: tag = (fieldNumber << 3) | wireType.
    private const int WireVarint = 0, WireFixed64 = 1, WireLengthDelimited = 2, WireFixed32 = 5;

    // ── MiniProtoWriter ───────────────────────────────────────────────────────

    /// <summary>Minimal protobuf wire-format writer — just enough to byte-build
    /// ModelProto payloads without any protobuf package. Nested messages are
    /// built with a child writer, then length-delimited into the parent.</summary>
    private sealed class MiniProtoWriter
    {
        private readonly List<byte> _bytes = [];

        public byte[] ToArray() => [.. _bytes];

        /// <summary>Raw bytes, no framing — fixed-width payloads and corrupt-input tests.</summary>
        public void WriteRaw(byte[] raw) => _bytes.AddRange(raw);

        public void WriteTag(int field, int wireType) => WriteVarint((ulong)((field << 3) | wireType));

        public void WriteVarint(ulong value)
        {
            while (value >= 0x80)
            {
                _bytes.Add((byte)(value | 0x80));
                value >>= 7;
            }
            _bytes.Add((byte)value);
        }

        /// <summary>Varint field. Negative int64 values take the 10-byte two's-complement form.</summary>
        public void WriteVarintField(int field, long value)
        {
            WriteTag(field, WireVarint);
            WriteVarint(unchecked((ulong)value));
        }

        public void WriteLengthDelimited(int field, byte[] payload)
        {
            WriteTag(field, WireLengthDelimited);
            WriteVarint((ulong)payload.Length);
            _bytes.AddRange(payload);
        }

        public void WriteString(int field, string value)
            => WriteLengthDelimited(field, Encoding.UTF8.GetBytes(value));

        public void WriteFixed32Float(int field, float value)
        {
            WriteTag(field, WireFixed32);
            _bytes.AddRange(BitConverter.GetBytes(value));       // IEEE-754 little-endian
        }

        /// <summary>Packed repeated varints: every element in one length-delimited blob.</summary>
        public void WritePackedVarints(int field, long[] values)
        {
            var child = new MiniProtoWriter();
            foreach (long v in values) child.WriteVarint(unchecked((ulong)v));
            WriteLengthDelimited(field, child.ToArray());
        }

        /// <summary>Packed repeated floats: little-endian fixed32s in one blob (float_data).</summary>
        public void WritePackedFloats(int field, float[] values)
        {
            byte[] payload = new byte[values.Length * 4];
            Buffer.BlockCopy(values, 0, payload, 0, payload.Length);
            WriteLengthDelimited(field, payload);
        }
    }

    // ── ModelProto builders (field numbers straight from onnx.proto) ─────────

    /// <summary>OperatorSetIdProto: domain=1 (string), version=2 (varint).</summary>
    private static byte[] Opset(long version, string domain = "")
    {
        var w = new MiniProtoWriter();
        if (domain.Length > 0) w.WriteString(1, domain);
        w.WriteVarintField(2, version);
        return w.ToArray();
    }

    /// <summary>ValueInfoProto: name=1, type=2 { tensor_type=1 { elem_type=1,
    /// shape=2 { dim=1 { dim_value=1 | dim_param=2 } } } }. A string dim emits
    /// dim_param (dynamic size), an int emits dim_value.</summary>
    private static byte[] ValueInfo(string name, params object[] dims)
    {
        var shape = new MiniProtoWriter();
        foreach (object d in dims)
        {
            var dim = new MiniProtoWriter();
            if (d is string p) dim.WriteString(2, p);            // dim_param
            else dim.WriteVarintField(1, Convert.ToInt64(d));    // dim_value
            shape.WriteLengthDelimited(1, dim.ToArray());
        }

        var tensorType = new MiniProtoWriter();
        tensorType.WriteVarintField(1, OnnxDataType.Float);      // elem_type
        tensorType.WriteLengthDelimited(2, shape.ToArray());     // shape

        var type = new MiniProtoWriter();
        type.WriteLengthDelimited(1, tensorType.ToArray());      // tensor_type

        var vi = new MiniProtoWriter();
        vi.WriteString(1, name);
        vi.WriteLengthDelimited(2, type.ToArray());
        return vi.ToArray();
    }

    /// <summary>NodeProto: input=1, output=2, name=3, op_type=4, attribute=5.</summary>
    private static byte[] Node(string opType, string name, string[] inputs, string[] outputs,
                               params byte[][] attributes)
    {
        var n = new MiniProtoWriter();
        foreach (string i in inputs) n.WriteString(1, i);
        foreach (string o in outputs) n.WriteString(2, o);
        n.WriteString(3, name);
        n.WriteString(4, opType);
        foreach (byte[] a in attributes) n.WriteLengthDelimited(5, a);
        return n.ToArray();
    }

    // AttributeProto: name=1, f=2 (fixed32), i=3, s=4, ints=8, type=20.

    private static byte[] AttrInt(string name, long value)
    {
        var a = new MiniProtoWriter();
        a.WriteString(1, name);
        a.WriteVarintField(3, value);                            // i
        a.WriteVarintField(20, (long)OnnxAttributeType.Int);
        return a.ToArray();
    }

    private static byte[] AttrFloat(string name, float value)
    {
        var a = new MiniProtoWriter();
        a.WriteString(1, name);
        a.WriteFixed32Float(2, value);                           // f
        a.WriteVarintField(20, (long)OnnxAttributeType.Float);
        return a.ToArray();
    }

    private static byte[] AttrString(string name, string value)
    {
        var a = new MiniProtoWriter();
        a.WriteString(1, name);
        a.WriteString(4, value);                                 // s (bytes)
        a.WriteVarintField(20, (long)OnnxAttributeType.String);
        return a.ToArray();
    }

    /// <summary>ints attribute — packed (one blob) or unpacked (one varint per element);
    /// both encodings are legal wire format for a repeated int64 field.</summary>
    private static byte[] AttrInts(string name, long[] values, bool packed)
    {
        var a = new MiniProtoWriter();
        a.WriteString(1, name);
        if (packed) a.WritePackedVarints(8, values);             // ints
        else foreach (long v in values) a.WriteVarintField(8, v);
        a.WriteVarintField(20, (long)OnnxAttributeType.Ints);
        return a.ToArray();
    }

    // TensorProto: dims=1, data_type=2, float_data=4 (packed), int64_data=7 (packed),
    //              name=8, raw_data=9.

    private static byte[] InitializerRawFloat(string name, long[] dims, float[] values)
    {
        var t = new MiniProtoWriter();
        t.WritePackedVarints(1, dims);
        t.WriteVarintField(2, OnnxDataType.Float);
        t.WriteString(8, name);
        byte[] raw = new byte[values.Length * 4];
        Buffer.BlockCopy(values, 0, raw, 0, raw.Length);         // IEEE-754 little-endian
        t.WriteLengthDelimited(9, raw);                          // raw_data
        return t.ToArray();
    }

    private static byte[] InitializerPackedFloat(string name, long[] dims, float[] values)
    {
        var t = new MiniProtoWriter();
        t.WritePackedVarints(1, dims);
        t.WriteVarintField(2, OnnxDataType.Float);
        t.WritePackedFloats(4, values);                          // float_data
        t.WriteString(8, name);
        return t.ToArray();
    }

    private static byte[] InitializerInt64(string name, long[] dims, long[] values)
    {
        var t = new MiniProtoWriter();
        t.WritePackedVarints(1, dims);
        t.WriteVarintField(2, OnnxDataType.Int64);
        t.WritePackedVarints(7, values);                         // int64_data
        t.WriteString(8, name);
        return t.ToArray();
    }

    /// <summary>GraphProto: node=1, name=2, initializer=5, input=11, output=12.</summary>
    private static byte[] Graph(string name, byte[][] nodes, byte[][] initializers,
                                byte[][] inputs, byte[][] outputs)
    {
        var g = new MiniProtoWriter();
        foreach (byte[] n in nodes) g.WriteLengthDelimited(1, n);
        g.WriteString(2, name);
        foreach (byte[] i in initializers) g.WriteLengthDelimited(5, i);
        foreach (byte[] i in inputs) g.WriteLengthDelimited(11, i);
        foreach (byte[] o in outputs) g.WriteLengthDelimited(12, o);
        return g.ToArray();
    }

    /// <summary>ModelProto: graph=7, opset_import=8.</summary>
    private static byte[] Model(byte[] graph, params byte[][] opsets)
    {
        var m = new MiniProtoWriter();
        m.WriteLengthDelimited(7, graph);
        foreach (byte[] o in opsets) m.WriteLengthDelimited(8, o);
        return m.ToArray();
    }

    // ── Initializers: raw_data, packed float_data, packed int64_data ─────────

    [Fact]
    public void Load_ParsesInitializers_FromRawDataAndTypedArrays()
    {
        byte[] graph = Graph(
            name: "weights-graph",
            nodes: [],
            initializers:
            [
                InitializerRawFloat("w_raw", [2, 2], [1.5f, -2f, 3.25f, 4f]),
                InitializerPackedFloat("w_packed", [3], [0.5f, 1.5f, 2.5f]),
                InitializerInt64("shape_i64", [2], [640, -3]),   // negative → 10-byte varint
            ],
            inputs: [],
            outputs: []);

        // A foreign-domain opset first: the ai.onnx one (empty domain) must win.
        var model = OnnxModel.Load(Model(graph, Opset(99, "com.custom"), Opset(13)));
        OnnxGraph g = model.Graph;

        Assert.Equal("weights-graph", g.Name);
        Assert.Equal(13L, g.OpsetVersion);
        Assert.Equal(3, g.Initializers.Count);

        OnnxTensorData raw = g.Initializers["w_raw"];
        Assert.Equal(OnnxDataType.Float, raw.DataType);
        Assert.Equal(new long[] { 2, 2 }, raw.Dims);
        Assert.NotNull(raw.Floats);
        Assert.Equal(new[] { 1.5f, -2f, 3.25f, 4f }, raw.Floats);

        OnnxTensorData packed = g.Initializers["w_packed"];
        Assert.Equal(new long[] { 3 }, packed.Dims);
        Assert.NotNull(packed.Floats);
        Assert.Equal(new[] { 0.5f, 1.5f, 2.5f }, packed.Floats);

        OnnxTensorData i64 = g.Initializers["shape_i64"];
        Assert.Equal(OnnxDataType.Int64, i64.DataType);
        Assert.Equal(new long[] { 2 }, i64.Dims);
        Assert.NotNull(i64.Longs);
        Assert.Equal(new long[] { 640, -3 }, i64.Longs);
    }

    // ── Node attributes ───────────────────────────────────────────────────────

    [Theory]
    [InlineData(true)]   // ints packed into one length-delimited blob …
    [InlineData(false)]  // … and unpacked as repeated varints — both are legal
    public void NodeAttributes_ParseWithBothIntsEncodings_AndFallBackWhenMissing(bool packed)
    {
        byte[] conv = Node("Conv", "conv1", ["x", "w"], ["y"],
            AttrInts("strides", [2, 2], packed),
            AttrInt("group", 2),
            AttrFloat("leak", 0.1f),
            AttrString("auto_pad", "SAME_UPPER"));

        var model = OnnxModel.Load(Model(Graph("attr-graph", [conv], [], [], []), Opset(13)));

        OnnxNode node = Assert.Single(model.Graph.Nodes);
        Assert.Equal("Conv", node.OpType);
        Assert.Equal("conv1", node.Name);
        Assert.Equal(new[] { "x", "w" }, node.Inputs);
        Assert.Equal(new[] { "y" }, node.Outputs);

        Assert.Equal(new long[] { 2, 2 }, node.GetInts("strides"));
        Assert.Equal(2L, node.GetInt("group"));
        Assert.Equal(0.1f, node.GetFloat("leak"));
        Assert.Equal("SAME_UPPER", node.GetString("auto_pad"));

        // Missing attribute names fall back to the caller-supplied defaults.
        Assert.Equal(1L, node.GetInt("group_missing", 1));
        Assert.Equal(new long[] { 1, 1 }, node.GetInts("pads_missing", [1, 1]));
        Assert.Empty(node.GetInts("also_missing"));
        Assert.Equal(2.5f, node.GetFloat("alpha_missing", 2.5f));
        Assert.Equal("NOTSET", node.GetString("mode_missing", "NOTSET"));
    }

    // ── Graph inputs vs initializers, dynamic dims ────────────────────────────

    [Fact]
    public void GraphInputs_ExcludeInitializers_AndDynamicDimsParseAsMinusOne()
    {
        byte[] graph = Graph(
            name: "io-graph",
            nodes: [],
            initializers: [InitializerRawFloat("w", [2], [1f, 2f])],
            inputs:
            [
                ValueInfo("x", "N", 3),   // dim_param "N" → dynamic → -1
                ValueInfo("w", 2),        // initializer listed as input (pre-IR-4 style) → excluded
            ],
            outputs: [ValueInfo("y", 1)]);

        var model = OnnxModel.Load(Model(graph, Opset(13)));

        OnnxValueInfo x = Assert.Single(model.Graph.Inputs);    // "w" filtered out
        Assert.Equal("x", x.Name);
        Assert.Equal(new[] { -1, 3 }, x.Shape);

        Assert.Equal(new[] { "x" }, model.InputNames);
        Assert.Equal(new[] { "y" }, model.OutputNames);
    }

    // ── Unknown fields ────────────────────────────────────────────────────────

    [Fact]
    public void UnknownFields_AtModelGraphAndNodeLevel_AreSkipped()
    {
        // Node with unknown fields sprinkled between the real ones.
        var n = new MiniProtoWriter();
        n.WriteString(1, "x");                                   // input
        n.WriteVarintField(99, 12345);                           // unknown varint
        n.WriteString(2, "y");                                   // output
        n.WriteLengthDelimited(98, [0xDE, 0xAD, 0xBE, 0xEF]);    // unknown length-delimited
        n.WriteString(4, "Relu");                                // op_type

        // Graph with unknown fields around the node.
        var g = new MiniProtoWriter();
        g.WriteVarintField(99, 7);
        g.WriteLengthDelimited(1, n.ToArray());                  // node
        g.WriteLengthDelimited(98, Encoding.UTF8.GetBytes("ignore me"));
        g.WriteString(2, "unknown-fields-graph");                // name
        g.WriteLengthDelimited(11, ValueInfo("x", 1));           // input
        g.WriteLengthDelimited(12, ValueInfo("y", 1));           // output

        // Model with an unknown field of every wire type the spec allows here:
        // varint, fixed64, length-delimited and fixed32 — all must be skipped.
        var m = new MiniProtoWriter();
        m.WriteVarintField(99, 7);
        m.WriteTag(97, WireFixed64);
        m.WriteRaw(BitConverter.GetBytes(0x0102030405060708UL));
        m.WriteLengthDelimited(7, g.ToArray());                  // graph
        m.WriteLengthDelimited(98, [1, 2, 3]);
        m.WriteTag(96, WireFixed32);
        m.WriteRaw(BitConverter.GetBytes(0xCAFEBABEu));
        m.WriteLengthDelimited(8, Opset(13));                    // opset_import

        var model = OnnxModel.Load(m.ToArray());

        Assert.Equal("unknown-fields-graph", model.Graph.Name);
        Assert.Equal(13L, model.Graph.OpsetVersion);
        OnnxNode node = Assert.Single(model.Graph.Nodes);
        Assert.Equal("Relu", node.OpType);
        Assert.Equal(new[] { "x" }, node.Inputs);
        Assert.Equal(new[] { "y" }, node.Outputs);
        Assert.Equal(new[] { "x" }, model.InputNames);
    }

    // ── End to end ────────────────────────────────────────────────────────────

    [Fact]
    public void EndToEnd_TinyAddModel_ParsesAndRuns()
    {
        byte[] graph = Graph(
            name: "add-graph",
            nodes: [Node("Add", "add0", ["x", "w"], ["y"])],
            initializers: [InitializerRawFloat("w", [1, 2], [10f, 20f])],
            inputs: [ValueInfo("x", 1, 2), ValueInfo("w", 1, 2)],  // "w" declared too → excluded
            outputs: [ValueInfo("y", 1, 2)]);

        var model = OnnxModel.Load(Model(graph, Opset(13)));

        Assert.Equal(new[] { "x" }, model.InputNames);           // the caller feeds only "x"
        Assert.DoesNotContain("w", model.InputNames);

        var outputs = model.Run(new() { ["x"] = new Tensor([1, 2], [1f, 2f]) });

        Tensor y = outputs["y"];
        Assert.Equal(new[] { 1, 2 }, y.Shape);
        Assert.Equal(new[] { 11f, 22f }, y.Data);
    }

    // ── Malformed input ───────────────────────────────────────────────────────

    [Fact]
    public void Load_LengthRunningPastBuffer_Throws()
    {
        var m = new MiniProtoWriter();
        m.WriteTag(7, WireLengthDelimited);   // ModelProto.graph …
        m.WriteVarint(1000);                  // … claims 1000 payload bytes …
        m.WriteRaw([1, 2, 3]);                // … but only 3 follow.

        Assert.ThrowsAny<Exception>(() => OnnxModel.Load(m.ToArray()));
    }

    [Fact]
    public void Load_TruncatedModel_Throws()
    {
        byte[] graph = Graph(
            name: "t",
            nodes: [Node("Add", "a", ["x", "x"], ["y"])],
            initializers: [InitializerRawFloat("w", [2], [1f, 2f])],
            inputs: [ValueInfo("x", 2)],
            outputs: [ValueInfo("y", 2)]);
        byte[] whole = Model(graph, Opset(13));

        // Cutting inside the graph message leaves its declared length past the end.
        byte[] truncated = whole[..(whole.Length / 2)];

        Assert.ThrowsAny<Exception>(() => OnnxModel.Load(truncated));
    }
}
