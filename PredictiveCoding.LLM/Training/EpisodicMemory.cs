using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;
using PredictiveCoding.Networks.Memory;

namespace PredictiveCoding.LLM.Training;

// ─────────────────────────────────────────────────────────────────────────────
// EpisodicMemory  —  the hippocampal graft (roadmap-Pro Phase 7).
//
// A text-payload episodic store on top of the core's associative MemoryBank
// (fixed-capacity, cosine top-k, FIFO eviction — the LM's hippocampus). Remember
// stores an embedding KEY with its source TEXT; Recall returns the texts nearest
// a query embedding. The vector index (MemoryBank) does the associative recall;
// a parallel ring of the keys+payloads is the durable episodic trace, so the
// whole memory saves and loads with the model.
//
// The bank stores a single float — the payload SLOT id — as its value; because
// this ring and the bank's ring share capacity and are written in lockstep, a
// live bank entry always maps to a live payload slot.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class EpisodicMemory
{
    private const uint Magic = 0x4550_4931;   // "EPI1"

    private readonly MemoryBank _bank;
    private readonly float[]?[] _keys;         // key mirror, for persistence
    private readonly string?[]  _payloads;
    private long _writes;

    public int KeyDim   => _bank.KeyDim;
    public int Capacity => _bank.Capacity;
    public int Count    => _bank.Count;

    public EpisodicMemory(int keyDim, int capacity = 1024)
    {
        if (keyDim < 1)    throw new ArgumentOutOfRangeException(nameof(keyDim));
        if (capacity < 1)  throw new ArgumentOutOfRangeException(nameof(capacity));
        _bank     = new MemoryBank(capacity, keyDim, valueDim: 1);
        _keys     = new float[]?[capacity];
        _payloads = new string?[capacity];
    }

    /// <summary>Stores <paramref name="text"/> under its embedding key (FIFO once full).</summary>
    public void Remember(float[] key, string text)
    {
        if (key.Length != KeyDim) throw new ArgumentException($"Key has {key.Length} dims; expected {KeyDim}.");
        ArgumentNullException.ThrowIfNull(text);

        int slot = (int)(_writes % Capacity);
        _keys[slot]     = (float[])key.Clone();
        _payloads[slot] = text;
        _bank.Write(key, [slot]);
        _writes++;
    }

    /// <summary>The <paramref name="topK"/> remembered texts nearest <paramref name="query"/> (cosine).</summary>
    public IReadOnlyList<(string text, float similarity)> Recall(float[] query, int topK)
    {
        var hits = _bank.QueryTopK(query, topK);
        var result = new List<(string, float)>(hits.Length);
        foreach (var h in hits)
        {
            int slot = (int)h.Value[0];
            if (_payloads[slot] is string s) result.Add((s, h.Similarity));
        }
        return result;
    }

    /// <summary>
    /// Le voci vive in ordine di età (la più vecchia prima). Il checkpoint le
    /// scrive UNA PER RECORD nello store FASTER: la memoria è una collezione che
    /// cresce, non uno snapshot monolitico — così la page resta piccola e si apre
    /// la strada alla persistenza incrementale (solo le voci nuove).
    /// </summary>
    public IEnumerable<(float[] Key, string Text)> LiveEntriesOldestFirst()
    {
        foreach (int slot in LiveSlotsOldestFirst())
            yield return (_keys[slot]!, _payloads[slot]!);
    }

    // ── Persistence ────────────────────────────────────────────────────────────

    /// <summary>Writes the live entries in age order so a reload reproduces the ring state.</summary>
    public void Save(string path)
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        File.WriteAllBytes(path, SaveToBytes());
    }

    /// <summary>
    /// Serializes the live entries (age order) to a byte array — the same layout
    /// <see cref="Save(string)"/> writes to disk, so it can be embedded inside a
    /// larger container (e.g. the whole-LLM FASTER store).
    /// </summary>
    public byte[] SaveToBytes()
    {
        using var ms = new MemoryStream();
        Span<byte> hdr = stackalloc byte[16];
        BinaryPrimitives.WriteUInt32LittleEndian(hdr[..4], Magic);
        BinaryPrimitives.WriteInt32LittleEndian(hdr[4..8], KeyDim);
        BinaryPrimitives.WriteInt32LittleEndian(hdr[8..12], Capacity);
        BinaryPrimitives.WriteInt32LittleEndian(hdr[12..16], Count);
        ms.Write(hdr);

        Span<byte> len = stackalloc byte[4];
        foreach (int slot in LiveSlotsOldestFirst())
        {
            ms.Write(MemoryMarshal.AsBytes(_keys[slot]!.AsSpan()));
            byte[] textBytes = Encoding.UTF8.GetBytes(_payloads[slot]!);
            BinaryPrimitives.WriteInt32LittleEndian(len, textBytes.Length);
            ms.Write(len);
            ms.Write(textBytes);
        }
        return ms.ToArray();
    }

    /// <summary>Reloads a memory saved by <see cref="Save(string)"/>; the key dimension must match.</summary>
    public static EpisodicMemory Load(string path, int expectedKeyDim)
        => LoadFromBytes(File.ReadAllBytes(path), expectedKeyDim);

    /// <summary>Reloads a memory from bytes produced by <see cref="SaveToBytes"/>; the key dimension must match.</summary>
    public static EpisodicMemory LoadFromBytes(byte[] bytes, int expectedKeyDim)
    {
        var span = bytes.AsSpan();
        if (BinaryPrimitives.ReadUInt32LittleEndian(span[..4]) != Magic)
            throw new InvalidDataException("Not an episodic-memory blob.");
        int keyDim   = BinaryPrimitives.ReadInt32LittleEndian(span[4..8]);
        int capacity = BinaryPrimitives.ReadInt32LittleEndian(span[8..12]);
        int count    = BinaryPrimitives.ReadInt32LittleEndian(span[12..16]);
        if (keyDim != expectedKeyDim)
            throw new InvalidDataException($"Memory key dim {keyDim} ≠ model embedding dim {expectedKeyDim}.");

        var mem = new EpisodicMemory(keyDim, capacity);
        int pos = 16;
        for (int i = 0; i < count; i++)
        {
            var key = new float[keyDim];
            span.Slice(pos, keyDim * 4).CopyTo(MemoryMarshal.AsBytes(key.AsSpan())); pos += keyDim * 4;
            int textLen = BinaryPrimitives.ReadInt32LittleEndian(span.Slice(pos, 4)); pos += 4;
            string text = Encoding.UTF8.GetString(span.Slice(pos, textLen)); pos += textLen;
            mem.Remember(key, text);
        }
        return mem;
    }

    private IEnumerable<int> LiveSlotsOldestFirst()
    {
        if (Count < Capacity)
            for (int i = 0; i < Count; i++) yield return i;             // never wrapped: 0..Count-1 in write order
        else
            for (int i = 0; i < Capacity; i++)
                yield return (int)((_writes + i) % Capacity);            // oldest is at _writes % Capacity
    }
}
