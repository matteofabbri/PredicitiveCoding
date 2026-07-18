using System.Runtime.InteropServices;
using System.Text;
using PredictiveCoding.Onnx.Proto;

namespace PredictiveCoding.Onnx;

// ─────────────────────────────────────────────────────────────────────────────
// PcnxFormat — the in-house on-disk model format ("PCNX" = PredictiveCoding
// oNNX). A direct little-endian binary dump of OnnxGraph: no protobuf, weights
// stored as raw float32/int64 blobs, so loading is a straight read into the
// exact structures the Executor walks.
//
// Round-trips everything OnnxParser materialises. Initializers whose data type
// the runtime cannot represent (fp16, int8, …) are written as dtype+dims with
// no payload — exactly like the parsed form, they fail lazily only if a Run
// actually touches them.
// ─────────────────────────────────────────────────────────────────────────────

public static class PcnxFormat
{
    /// <summary>Suggested file extension for saved models.</summary>
    public const string Extension = ".pcnx";

    private const int Version = 1;
    private static ReadOnlySpan<byte> Magic => "PCNX"u8;

    private const byte PayloadNone = 0;    // dtype/dims only (unsupported dtype carried through)
    private const byte PayloadFloats = 1;  // raw float32, little-endian
    private const byte PayloadLongs = 2;   // raw int64, little-endian

    /// <summary>True when <paramref name="bytes"/> starts with the PCNX magic.</summary>
    public static bool LooksLikePcnx(ReadOnlySpan<byte> bytes)
        => bytes.Length >= Magic.Length && bytes[..Magic.Length].SequenceEqual(Magic);

    // ── Writing ─────────────────────────────────────────────────────────────────

    // Written to a temp file and moved into place on success, so a mid-write
    // failure (unsupported attribute, disk full) can never truncate or corrupt
    // an existing model at the destination.
    public static void Write(string path, OnnxGraph graph)
    {
        string tmp = path + ".tmp";
        try
        {
            using (FileStream fs = File.Create(tmp))
                Write(fs, graph);
            File.Move(tmp, path, overwrite: true);
        }
        catch
        {
            try { File.Delete(tmp); } catch { /* best effort */ }
            throw;
        }
    }

    public static void Write(Stream stream, OnnxGraph graph)
    {
        ArgumentNullException.ThrowIfNull(graph);
        using var w = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        w.Write(Magic);
        w.Write(Version);
        w.Write(graph.OpsetVersion);
        w.Write(graph.Name);

        WriteValueInfos(w, graph.Inputs);
        WriteValueInfos(w, graph.Outputs);

        w.Write(graph.Initializers.Count);
        foreach ((string name, OnnxTensorData tensor) in graph.Initializers)
        {
            w.Write(name);
            WriteTensorData(w, tensor);
        }

        w.Write(graph.Nodes.Count);
        foreach (OnnxNode node in graph.Nodes)
            WriteNode(w, node);
    }

    private static void WriteValueInfos(BinaryWriter w, List<OnnxValueInfo> infos)
    {
        w.Write(infos.Count);
        foreach (OnnxValueInfo info in infos)
        {
            w.Write(info.Name);
            w.Write(info.Shape.Length);
            foreach (int d in info.Shape) w.Write(d);
        }
    }

    private static void WriteTensorData(BinaryWriter w, OnnxTensorData t)
    {
        w.Write(t.Name);
        w.Write(t.DataType);
        w.Write(t.Dims.Length);
        foreach (long d in t.Dims) w.Write(d);

        if (t.Floats is not null)
        {
            w.Write(PayloadFloats);
            w.Write(t.Floats.Length);
            w.Write(MemoryMarshal.AsBytes(t.Floats.AsSpan()));
        }
        else if (t.Longs is not null)
        {
            w.Write(PayloadLongs);
            w.Write(t.Longs.Length);
            w.Write(MemoryMarshal.AsBytes(t.Longs.AsSpan()));
        }
        else
        {
            w.Write(PayloadNone);
        }
    }

    private static void WriteNode(BinaryWriter w, OnnxNode node)
    {
        w.Write(node.Name);
        w.Write(node.OpType);

        w.Write(node.Inputs.Length);
        foreach (string s in node.Inputs) w.Write(s);
        w.Write(node.Outputs.Length);
        foreach (string s in node.Outputs) w.Write(s);

        w.Write(node.Attributes.Count);
        foreach (OnnxAttribute a in node.Attributes.Values)
        {
            w.Write(a.Name);
            w.Write((byte)a.Type);
            switch (a.Type)
            {
                case OnnxAttributeType.Float: w.Write(a.F); break;
                case OnnxAttributeType.Int: w.Write(a.I); break;
                case OnnxAttributeType.String:
                    // Null and "" are distinct: GetString falls back only on null,
                    // so coercing null → "" would change attribute lookups after a
                    // round-trip. Same for the tensor payload below.
                    w.Write(a.S is not null);
                    if (a.S is not null) w.Write(a.S);
                    break;
                case OnnxAttributeType.Floats:
                    float[] floats = a.Floats ?? [];
                    w.Write(floats.Length);
                    w.Write(MemoryMarshal.AsBytes(floats.AsSpan()));
                    break;
                case OnnxAttributeType.Ints:
                    long[] ints = a.Ints ?? [];
                    w.Write(ints.Length);
                    w.Write(MemoryMarshal.AsBytes(ints.AsSpan()));
                    break;
                case OnnxAttributeType.Strings:
                    string[] strings = a.Strings ?? [];
                    w.Write(strings.Length);
                    foreach (string s in strings) w.Write(s);
                    break;
                case OnnxAttributeType.Tensor:
                    w.Write(a.T is not null);
                    if (a.T is not null) WriteTensorData(w, a.T);
                    break;
                default:
                    throw new NotSupportedException(
                        $"Attribute '{a.Name}' of type {a.Type} (node '{node.Name}') cannot be serialized to PCNX.");
            }
        }
    }

    // ── Reading ─────────────────────────────────────────────────────────────────

    public static OnnxGraph Read(byte[] bytes)
    {
        using var ms = new MemoryStream(bytes, writable: false);
        return Read(ms);
    }

    public static OnnxGraph Read(Stream stream)
    {
        using var r = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);
        try
        {
            // Read(span) may legally return fewer bytes than asked on streaming
            // sources (pipes, decompression) — insist on all four magic bytes.
            Span<byte> magic = stackalloc byte[Magic.Length];
            if (stream.ReadAtLeast(magic, magic.Length, throwOnEndOfStream: false) != magic.Length ||
                !LooksLikePcnx(magic))
                throw new InvalidDataException("Not a PCNX model file (bad magic).");
            int version = r.ReadInt32();
            if (version != Version)
                throw new InvalidDataException($"PCNX version {version} is not supported (expected {Version}).");

            var graph = new OnnxGraph
            {
                OpsetVersion = r.ReadInt64(),
                Name = r.ReadString(),
                Inputs = ReadValueInfos(r),
                Outputs = ReadValueInfos(r),
            };

            int initCount = ReadCount(r, "initializer");
            for (int i = 0; i < initCount; i++)
            {
                string name = r.ReadString();
                graph.Initializers[name] = ReadTensorData(r);
            }

            int nodeCount = ReadCount(r, "node");
            graph.Nodes.Capacity = nodeCount;
            for (int i = 0; i < nodeCount; i++)
                graph.Nodes.Add(ReadNode(r));

            return graph;
        }
        catch (EndOfStreamException e)
        {
            throw new InvalidDataException("PCNX file is truncated.", e);
        }
    }

    /// <summary>Reads a count field, rejecting values a valid file cannot contain:
    /// negative, or claiming more elements than the stream has bytes left (each
    /// element occupies at least <paramref name="elementSize"/> bytes). Keeps a
    /// corrupt file from triggering huge allocations or confusing exceptions.</summary>
    private static int ReadCount(BinaryReader r, string what, int elementSize = 1)
    {
        int count = r.ReadInt32();
        if (count < 0)
            throw new InvalidDataException($"Corrupt PCNX file: negative {what} count ({count}).");
        Stream s = r.BaseStream;
        if (s.CanSeek && (long)count * elementSize > s.Length - s.Position)
            throw new InvalidDataException(
                $"Corrupt PCNX file: {what} count {count} exceeds the remaining file size.");
        return count;
    }

    private static List<OnnxValueInfo> ReadValueInfos(BinaryReader r)
    {
        int count = ReadCount(r, "value-info");
        var infos = new List<OnnxValueInfo>(count);
        for (int i = 0; i < count; i++)
        {
            var info = new OnnxValueInfo { Name = r.ReadString(), Shape = new int[ReadCount(r, "shape", sizeof(int))] };
            for (int d = 0; d < info.Shape.Length; d++) info.Shape[d] = r.ReadInt32();
            infos.Add(info);
        }
        return infos;
    }

    private static OnnxTensorData ReadTensorData(BinaryReader r)
    {
        var t = new OnnxTensorData { Name = r.ReadString(), DataType = r.ReadInt32() };
        t.Dims = new long[ReadCount(r, "tensor rank", sizeof(long))];
        for (int i = 0; i < t.Dims.Length; i++) t.Dims[i] = r.ReadInt64();

        byte payload = r.ReadByte();
        switch (payload)
        {
            case PayloadFloats: t.Floats = ReadFloats(r, ReadCount(r, "float payload", sizeof(float))); break;
            case PayloadLongs: t.Longs = ReadLongs(r, ReadCount(r, "int64 payload", sizeof(long))); break;
            case PayloadNone: break;
            default: throw new InvalidDataException($"Unknown PCNX tensor payload kind {payload}.");
        }
        return t;
    }

    private static OnnxNode ReadNode(BinaryReader r)
    {
        var node = new OnnxNode { Name = r.ReadString(), OpType = r.ReadString() };

        node.Inputs = new string[ReadCount(r, "node input")];
        for (int i = 0; i < node.Inputs.Length; i++) node.Inputs[i] = r.ReadString();
        node.Outputs = new string[ReadCount(r, "node output")];
        for (int i = 0; i < node.Outputs.Length; i++) node.Outputs[i] = r.ReadString();

        int attrCount = ReadCount(r, "attribute");
        for (int i = 0; i < attrCount; i++)
        {
            var a = new OnnxAttribute { Name = r.ReadString(), Type = (OnnxAttributeType)r.ReadByte() };
            switch (a.Type)
            {
                case OnnxAttributeType.Float: a.F = r.ReadSingle(); break;
                case OnnxAttributeType.Int: a.I = r.ReadInt64(); break;
                case OnnxAttributeType.String: a.S = r.ReadBoolean() ? r.ReadString() : null; break;
                case OnnxAttributeType.Floats: a.Floats = ReadFloats(r, ReadCount(r, "floats attribute", sizeof(float))); break;
                case OnnxAttributeType.Ints: a.Ints = ReadLongs(r, ReadCount(r, "ints attribute", sizeof(long))); break;
                case OnnxAttributeType.Strings:
                    a.Strings = new string[ReadCount(r, "strings attribute")];
                    for (int s = 0; s < a.Strings.Length; s++) a.Strings[s] = r.ReadString();
                    break;
                case OnnxAttributeType.Tensor: a.T = r.ReadBoolean() ? ReadTensorData(r) : null; break;
                default: throw new InvalidDataException($"Unknown PCNX attribute type {(byte)a.Type}.");
            }
            node.Attributes[a.Name] = a;
        }
        return node;
    }

    // BinaryReader does not buffer past what it returns, so bulk payloads can be
    // read straight from the underlying stream into the target arrays.
    private static float[] ReadFloats(BinaryReader r, int count)
    {
        var data = new float[count];
        r.BaseStream.ReadExactly(MemoryMarshal.AsBytes(data.AsSpan()));
        return data;
    }

    private static long[] ReadLongs(BinaryReader r, int count)
    {
        var data = new long[count];
        r.BaseStream.ReadExactly(MemoryMarshal.AsBytes(data.AsSpan()));
        return data;
    }
}
