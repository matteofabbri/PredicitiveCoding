using System.Buffers.Binary;
using System.Text;

namespace PredictiveCoding.Onnx.Proto;

// ─────────────────────────────────────────────────────────────────────────────
// ProtoReader — a minimal protobuf wire-format reader, just enough for ONNX.
//
// A mutable struct over (buffer, position, end): ReadTag/ReadVarint/Read* move
// the position forward, ReadSubReader hands out an independent reader scoped to
// a nested message's byte range — no copying, no allocation beyond the caller's
// own materialisation. Only the four wire types ONNX files use are implemented
// (varint, fixed64, length-delimited, fixed32); the long-deprecated group wire
// types throw NotSupportedException. Every read validates against the end
// offset first, so malformed input surfaces as InvalidDataException with the
// failing offset instead of reading out of bounds.
// ─────────────────────────────────────────────────────────────────────────────

/// <summary>Sequential reader over a protobuf-encoded byte range.</summary>
public struct ProtoReader
{
    /// <summary>Wire type 0: base-128 varint.</summary>
    public const int WireVarint = 0;

    /// <summary>Wire type 1: 8-byte little-endian value.</summary>
    public const int WireFixed64 = 1;

    /// <summary>Wire type 2: length-prefixed bytes (nested message, string, bytes, packed repeated).</summary>
    public const int WireLengthDelimited = 2;

    /// <summary>Wire type 5: 4-byte little-endian value.</summary>
    public const int WireFixed32 = 5;

    private readonly byte[] _buf;
    private readonly int _end;
    private int _pos;

    /// <summary>Reader over an entire buffer.</summary>
    public ProtoReader(byte[] buffer) : this(buffer, 0, buffer.Length) { }

    /// <summary>Reader over the byte range [offset, offset + length) of <paramref name="buffer"/>.</summary>
    public ProtoReader(byte[] buffer, int offset, int length)
    {
        if ((uint)offset > (uint)buffer.Length || (uint)length > (uint)(buffer.Length - offset))
            throw new ArgumentOutOfRangeException(nameof(length),
                $"Range [{offset}, {offset}+{length}) lies outside the {buffer.Length}-byte buffer.");
        _buf = buffer;
        _pos = offset;
        _end = offset + length;
    }

    /// <summary>True when every byte of this reader's range has been consumed —
    /// the loop condition for "read fields until the message ends".</summary>
    public readonly bool AtEnd => _pos >= _end;

    /// <summary>Current absolute offset into the underlying buffer (for diagnostics).</summary>
    public readonly int Position => _pos;

    /// <summary>Read a field key: returns the field number and the wire type (0, 1, 2 or 5).</summary>
    public (int FieldNumber, int WireType) ReadTag()
    {
        int start = _pos;
        ulong key = ReadVarint();
        int wireType = (int)(key & 0x7);
        ulong fieldNumber = key >> 3;
        if (fieldNumber == 0 || fieldNumber > int.MaxValue)
            throw new InvalidDataException($"Invalid protobuf field number {fieldNumber} at offset {start}.");
        return ((int)fieldNumber, wireType);
    }

    /// <summary>Read a little-endian base-128 varint (at most 10 bytes for 64 bits).
    /// Signed protobuf values are two's complement: cast the result to long.</summary>
    public ulong ReadVarint()
    {
        int start = _pos;
        ulong result = 0;
        for (int shift = 0; shift < 70; shift += 7)
        {
            if (_pos >= _end)
                throw new InvalidDataException($"Truncated varint at offset {start}: buffer ends mid-value.");
            byte b = _buf[_pos++];
            result |= (ulong)(b & 0x7Fu) << shift;
            if ((b & 0x80) == 0) return result;
        }
        throw new InvalidDataException($"Malformed varint at offset {start}: longer than 10 bytes.");
    }

    /// <summary>Read the length prefix of a wire-type-2 field and consume its payload,
    /// returning the payload's byte range in the underlying buffer.</summary>
    public (int Offset, int Length) ReadLengthDelimited()
    {
        int start = _pos;
        ulong length = ReadVarint();
        if (length > (ulong)(_end - _pos))
            throw new InvalidDataException(
                $"Length-delimited field at offset {start} declares {length} bytes but only {_end - _pos} remain.");
        int offset = _pos;
        _pos += (int)length;
        return (offset, (int)length);
    }

    /// <summary>Read a length-delimited field as a span over the underlying buffer (no copy).</summary>
    public ReadOnlySpan<byte> ReadBytesSpan()
    {
        var (offset, length) = ReadLengthDelimited();
        return _buf.AsSpan(offset, length);
    }

    /// <summary>Read a length-delimited field as a fresh byte array (for payloads that outlive the buffer walk).</summary>
    public byte[] ReadBytes() => ReadBytesSpan().ToArray();

    /// <summary>Read a length-delimited field as a UTF-8 string.</summary>
    public string ReadString() => Encoding.UTF8.GetString(ReadBytesSpan());

    /// <summary>Read a length-delimited field as an independent reader over the nested
    /// message (or packed repeated block); this reader skips past it.</summary>
    public ProtoReader ReadSubReader()
    {
        var (offset, length) = ReadLengthDelimited();
        return new ProtoReader(_buf, offset, length);
    }

    /// <summary>Read a 4-byte little-endian value (wire type 5).</summary>
    public uint ReadFixed32()
    {
        if (_end - _pos < 4)
            throw new InvalidDataException($"Truncated fixed32 at offset {_pos}: needs 4 bytes, {_end - _pos} remain.");
        uint value = BinaryPrimitives.ReadUInt32LittleEndian(_buf.AsSpan(_pos, 4));
        _pos += 4;
        return value;
    }

    /// <summary>Read an 8-byte little-endian value (wire type 1).</summary>
    public ulong ReadFixed64()
    {
        if (_end - _pos < 8)
            throw new InvalidDataException($"Truncated fixed64 at offset {_pos}: needs 8 bytes, {_end - _pos} remain.");
        ulong value = BinaryPrimitives.ReadUInt64LittleEndian(_buf.AsSpan(_pos, 8));
        _pos += 8;
        return value;
    }

    /// <summary>Read a protobuf float: fixed32 bits reinterpreted as IEEE-754 single.</summary>
    public float ReadFloat() => BitConverter.UInt32BitsToSingle(ReadFixed32());

    /// <summary>Skip the payload of a field whose number we do not care about.
    /// Handles all four supported wire types; groups (3/4) throw NotSupportedException.</summary>
    public void SkipField(int wireType)
    {
        switch (wireType)
        {
            case WireVarint:          ReadVarint(); break;
            case WireFixed64:         Advance(8); break;
            case WireLengthDelimited: ReadLengthDelimited(); break;
            case WireFixed32:         Advance(4); break;
            case 3 or 4:
                throw new NotSupportedException(
                    $"Protobuf group wire type {wireType} at offset {_pos} is not supported (ONNX never uses groups).");
            default:
                throw new InvalidDataException($"Unknown protobuf wire type {wireType} at offset {_pos}.");
        }
    }

    // Move past count bytes, validating against the end offset.
    private void Advance(int count)
    {
        if (_end - _pos < count)
            throw new InvalidDataException($"Truncated field at offset {_pos}: needs {count} bytes, {_end - _pos} remain.");
        _pos += count;
    }
}
