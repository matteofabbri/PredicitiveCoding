using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using FASTER.core;

namespace PredictiveCoding.Model;

// ─────────────────────────────────────────────────────────────────────────────
// LayerStore
//
// FASTER KV store that maps layer index (int) → serialized layer state (SpanByte).
//
// Key   : Layer.Index  (int)
// Value : flat binary blob produced by LayerSerializer  (SpanByte)
//
// Usage:
//   using var store = new LayerStore();      // in-memory (ephemeral)
//   using var store = new LayerStore(path);  // log-backed (persistent)
//
//   store.WriteAll(network.Layers);          // checkpoint
//   store.ReadAll(network.Layers);           // restore
// ─────────────────────────────────────────────────────────────────────────────

internal sealed class LayerStore : IDisposable
{
    private readonly FasterKV<int, SpanByte> _store;
    private readonly IDevice _device;

    // SpanByteFunctions<TKey, TOutput, TContext>:  Input=SpanByte, Output=SpanByteAndMemory
    private readonly ClientSession<
        int, SpanByte,
        SpanByte, SpanByteAndMemory,
        Empty, SpanByteFunctions<int, SpanByteAndMemory, Empty>> _session;

    // ── Construction ──────────────────────────────────────────────────────────

    /// <param name="storagePath">
    /// Directory for the log device.  When null the store uses an ephemeral
    /// temp file that is deleted on disposal.
    /// </param>
    public LayerStore(string? storagePath = null)
    {
        bool ephemeral = storagePath is null;
        string dir = storagePath ?? Path.Combine(Path.GetTempPath(), $"faster_layers_{Guid.NewGuid():N}");
        Directory.CreateDirectory(dir);

        _device = Devices.CreateLogDevice(
            Path.Combine(dir, "layers.log"),
            preallocateFile: false,
            deleteOnClose: ephemeral);

        _store = new FasterKV<int, SpanByte>(
            size: 1L << 20,
            logSettings: new LogSettings
            {
                LogDevice      = _device,
                PageSizeBits   = 22,   // 4 MB pages
                MemorySizeBits = 26,   // 64 MB in memory
            });

        _session = _store
            .For(new SpanByteFunctions<int, SpanByteAndMemory, Empty>())
            .NewSession<SpanByteFunctions<int, SpanByteAndMemory, Empty>>();
    }

    // ── Write ─────────────────────────────────────────────────────────────────

    /// <summary>Serializes <paramref name="layer"/> and upserts it by index.</summary>
    public void Write(Layer layer)
    {
        int    key     = layer.Index;
        byte[] payload = LayerSerializer.Serialize(layer);

        // SpanByte on-disk layout: [int32 length][byte... data].
        // We prefix the serialized payload with its 4-byte length so that
        // SpanByte.Reinterpret can interpret the buffer correctly.
        byte[] buf = new byte[sizeof(int) + payload.Length];
        Unsafe.WriteUnaligned(ref buf[0], payload.Length);
        payload.CopyTo(buf, sizeof(int));

        // Pin the buffer for the duration of the FASTER call so the GC
        // cannot relocate it while FASTER is reading the pointer.
        var handle = GCHandle.Alloc(buf, GCHandleType.Pinned);
        try
        {
            ref SpanByte sb = ref SpanByte.Reinterpret(ref buf[0]);
            _session.Upsert(ref key, ref sb);
        }
        finally
        {
            handle.Free();
        }
    }

    /// <summary>Checkpoints every layer in <paramref name="layers"/>.</summary>
    public void WriteAll(IReadOnlyList<Layer> layers)
    {
        foreach (var layer in layers)
            Write(layer);
    }

    // ── Read ──────────────────────────────────────────────────────────────────

    /// <summary>
    /// Reads the stored state for <paramref name="index"/> and writes it into
    /// <paramref name="target"/>.  Returns false if no entry exists.
    /// </summary>
    public bool TryRead(int index, Layer target)
    {
        SpanByteAndMemory output = default;
        Status status = _session.Read(ref index, ref output);

        if (!status.Found) return false;

        try
        {
            ReadOnlySpan<byte> bytes = output.IsSpanByte
                ? output.SpanByte.AsReadOnlySpan()
                : output.Memory!.Memory.Span[..output.Length];

            LayerSerializer.Deserialize(bytes, target);
            return true;
        }
        finally
        {
            if (!output.IsSpanByte)
                output.Memory?.Dispose();
        }
    }

    /// <summary>
    /// Restores every layer in <paramref name="layers"/> from the store.
    /// Throws <see cref="KeyNotFoundException"/> if any layer index is missing.
    /// </summary>
    public void ReadAll(IReadOnlyList<Layer> layers)
    {
        foreach (var layer in layers)
        {
            if (!TryRead(layer.Index, layer))
                throw new KeyNotFoundException(
                    $"No checkpoint found for layer index {layer.Index}.");
        }
    }

    // ── Persistence ───────────────────────────────────────────────────────────

    /// <summary>Forces all pending log writes to the device.</summary>
    public void Flush() => _store.Log.FlushAndEvict(wait: true);

    // ── Disposal ──────────────────────────────────────────────────────────────

    public void Dispose()
    {
        _session.Dispose();
        _store.Dispose();
        _device.Dispose();
    }
}
