using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using FASTER.core;
using PredictiveCoding.Networks;

namespace PredictiveCoding.Model;

// ─────────────────────────────────────────────────────────────────────────────
// LayerStore
//
// FASTER KV store that maps layer index (int) → serialized layer state (SpanByte).
//
// Key   : Layer.Index  (int)
// Value : flat binary blob produced by LayerSerializer, stored as a SpanByte
//         whose 4-byte length header precedes the payload.
//
// Reads go through FasterKV. Because the key is a plain int (not a SpanByte),
// the stock SpanByteFunctions<Key,Output,Context> cannot be specialized to a
// SpanByte value and therefore ships NO read callback — its inherited
// SingleReader/ConcurrentReader return without populating the output. We supply
// LayerStoreFunctions, which overrides both to copy the stored value into the
// SpanByteAndMemory output (exactly what the built-in SpanByteFunctions<Context>
// does for SpanByte keys). FasterKV is the single source of truth for reads.
//
// Lazy persistence (opt-in): live Layer objects are kept in _resident so hot
// ReadLayer/Write calls avoid per-sample serialization (~6 MB arrays for large
// weight matrices, e.g. CIFAR-10); serialization + FasterKV writes are deferred
// to Flush(). This is a performance cache, not a correctness crutch — eager
// mode reads straight from FasterKV.
//
// Persistence/recovery: a full checkpoint is taken on Dispose. A store opened on
// a path that already holds a checkpoint is recovered instead of reinitialized.
//
// Thread safety: _resident and _dirtyKeys are guarded by Network's
// ReaderWriterLockSlim (all callers hold it before entering LayerStore);
// FasterKV sessions are per-thread (ThreadLocal).
//
// Usage:
//   using var store = new LayerStore(config, rng);
//   store.Write(index, layer);               // checkpoint a single layer
//   var layer = store.ReadLayer(index);      // restore a single layer
// ─────────────────────────────────────────────────────────────────────────────

internal sealed class LayerStore : IDisposable
{
    // FasterKV functions for an int key + SpanByte value. The base
    // SpanByteFunctions<Key,Output,Context> only implements the write/RMW
    // callbacks; we add the read callbacks so SpanByteAndMemory is populated.
    private sealed class LayerStoreFunctions : SpanByteFunctions<int, SpanByteAndMemory, Empty>
    {
        public override bool SingleReader(ref int key, ref SpanByte input, ref SpanByte value,
            ref SpanByteAndMemory dst, ref ReadInfo readInfo)
        {
            value.CopyTo(ref dst, MemoryPool<byte>.Shared);
            return true;
        }

        public override bool ConcurrentReader(ref int key, ref SpanByte input, ref SpanByte value,
            ref SpanByteAndMemory dst, ref ReadInfo readInfo)
        {
            value.CopyTo(ref dst, MemoryPool<byte>.Shared);
            return true;
        }
    }

    private readonly FasterKV<int, SpanByte> _store;
    private readonly IDevice _device;
    private readonly NetworkConfig Configuration;
    private readonly Random Random;
    private readonly string _checkpointDir;

    // FasterKV is thread-safe at the store level; sessions are NOT.
    // Each thread gets its own session created on first use via ThreadLocal.
    private readonly ThreadLocal<ClientSession<
        int, SpanByte, SpanByte, SpanByteAndMemory, Empty, LayerStoreFunctions>> _sessions;

    private ClientSession<int, SpanByte, SpanByte, SpanByteAndMemory, Empty, LayerStoreFunctions>
        Session => _sessions.Value!;

    private readonly bool _lazyPersistence;
    private readonly HashSet<int> _dirtyKeys = [];

    // In lazy mode, Layer objects are kept in memory so that ReadLayer can
    // return a clone without byte[] serialization/deserialization overhead.
    private readonly Dictionary<int, Layer>? _resident;

    private int _disposed;

    // ── Construction ──────────────────────────────────────────────────────────

    public LayerStore(NetworkConfig config, Random random, bool lazyPersistence = false)
    {
        Configuration    = config;
        Random           = random;
        _lazyPersistence = lazyPersistence;
        if (_lazyPersistence) _resident = [];

        Directory.CreateDirectory(config.Path);
        _checkpointDir = Path.Combine(config.Path, "checkpoints");

        _device = Devices.CreateLogDevice(
            Path.Combine(config.Path, "layers.log"),
            preallocateFile: false,
            deleteOnClose: false);

        int pageSizeBits   = ComputePageSizeBits(config);
        int memorySizeBits = pageSizeBits + 4;   // 16 pages in memory

        _store = new FasterKV<int, SpanByte>(
            size: 1L << 20,
            logSettings: new LogSettings
            {
                LogDevice      = _device,
                PageSizeBits   = pageSizeBits,
                MemorySizeBits = memorySizeBits,
            },
            checkpointSettings: new CheckpointSettings { CheckpointDir = _checkpointDir });

        _sessions = new ThreadLocal<ClientSession<
                int, SpanByte, SpanByte, SpanByteAndMemory, Empty, LayerStoreFunctions>>(
            () => _store
                .For(new LayerStoreFunctions())
                .NewSession<LayerStoreFunctions>(),
            trackAllValues: true);

        if (HasCheckpoint(_checkpointDir))
        {
            _store.Recover();
            if (_lazyPersistence) PopulateResidentFromStore();
            return;
        }

        // Fresh storage — initialize and persist every layer.
        for (int index = 0; index < Configuration.LayerSizes.Length; index++)
        {
            Layer layer = SeedLayer(index);
            if (_lazyPersistence) _resident![index] = layer;
        }
    }

    // True when CheckpointDir already holds at least one completed checkpoint.
    private static bool HasCheckpoint(string checkpointDir)
    {
        string cpr = Path.Combine(checkpointDir, "cpr-checkpoints");
        return Directory.Exists(cpr) && Directory.EnumerateDirectories(cpr).Any();
    }

    // After Recover(), rebuild the resident cache (lazy mode only) from the store.
    private void PopulateResidentFromStore()
    {
        for (int index = 0; index < Configuration.LayerSizes.Length; index++)
        {
            byte[]? payload = ReadFromStore(index);
            if (payload is not null)
                _resident![index] = LayerSerializer.AllocateAndDeserialize(
                    payload, index, Configuration.Activation);
        }
    }

    private Layer Initialize(int index)
    {
        int lowerSize = index > 0 ? Configuration.LayerSizes[index - 1] : 0;
        var layer = new Layer
        {
            Index              = index,
            Size               = Configuration.LayerSizes[index],
            LowerSize          = lowerSize,
            StateLearningRate  = Configuration.StateLearningRate,
            WeightLearningRate = Configuration.WeightLearningRate,
            WeightDecay        = Configuration.WeightDecay,
            ActivationType     = Configuration.Activation,
        };
        layer.Initialize(Random);
        if (Configuration.UseDiagonalPrecision) layer.InitDiagonalPrecision();
        return layer;
    }

    // Creates a fresh layer, persists it to the store and returns it.
    private Layer SeedLayer(int index)
    {
        Layer layer = Initialize(index);
        PersistToStore(index, LayerSerializer.Serialize(layer));
        return layer;
    }

    private void PersistToStore(int index, byte[] payload)
    {
        byte[] buf = new byte[sizeof(int) + payload.Length];
        Unsafe.WriteUnaligned(ref buf[0], payload.Length);
        payload.CopyTo(buf, sizeof(int));

        var handle = GCHandle.Alloc(buf, GCHandleType.Pinned);
        try
        {
            ref SpanByte sb = ref SpanByte.Reinterpret(ref buf[0]);
            Session.Upsert(ref index, ref sb);
        }
        finally
        {
            handle.Free();
        }
    }

    // Reads the serialized payload for a layer from FasterKV, or null if absent.
    private byte[]? ReadFromStore(int index)
    {
        SpanByteAndMemory output = default;
        Status status = Session.Read(ref index, ref output);

        if (status.IsPending)
        {
            Session.CompletePendingWithOutputs(out var iter, wait: true);
            using (iter)
            {
                if (iter.Next())
                {
                    status = iter.Current.Status;
                    output = iter.Current.Output;
                }
            }
        }

        if (!status.Found)
            return null;

        try
        {
            ReadOnlySpan<byte> bytes = output.IsSpanByte
                ? output.SpanByte.AsReadOnlySpan()
                : output.Memory!.Memory.Span[..output.Length];
            return bytes.ToArray();
        }
        finally
        {
            if (!output.IsSpanByte) output.Memory?.Dispose();
        }
    }

    // ── Factory read ─────────────────────────────────────────────────────────

    /// <summary>
    /// Returns a fresh <see cref="Layer"/> for <paramref name="index"/>.
    /// In lazy mode: clones the resident Layer object (no byte[] round-trip).
    /// In eager mode: deserializes from FasterKV.
    /// </summary>
    public Layer ReadLayer(int index)
    {
        if (_lazyPersistence)
        {
            if (!_resident!.TryGetValue(index, out Layer? resident))
            {
                byte[]? payload = ReadFromStore(index);
                resident = payload is not null
                    ? LayerSerializer.AllocateAndDeserialize(payload, index, Configuration.Activation)
                    : SeedLayer(index);
                _resident[index] = resident;
            }
            return resident.Clone();
        }

        byte[]? blob = ReadFromStore(index);
        return blob is not null
            ? LayerSerializer.AllocateAndDeserialize(blob, index, Configuration.Activation)
            : SeedLayer(index);
    }

    // ── Write ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Persists <paramref name="layer"/> to the store.
    /// Lazy mode: keeps the live Layer object and defers serialization +
    ///            FasterKV write until <see cref="Flush"/>.
    /// Eager mode: serializes immediately and upserts to FasterKV.
    /// </summary>
    public void Write(int index, Layer layer)
    {
        if (_lazyPersistence)
        {
            _resident![index] = layer;   // keep reference — no serialization
            _dirtyKeys.Add(index);
        }
        else
        {
            PersistToStore(index, LayerSerializer.Serialize(layer));
        }
    }

    /// <summary>
    /// Flushes dirty resident layers to FasterKV (no-op in eager mode), then
    /// forces the device log. Call once per epoch when
    /// <see cref="NetworkConfig.LazyPersistence"/> is true.
    /// </summary>
    public void Flush()
    {
        if (_lazyPersistence)
        {
            foreach (int idx in _dirtyKeys)
                PersistToStore(idx, LayerSerializer.Serialize(_resident![idx]));
            _dirtyKeys.Clear();
        }
        _store.Log.FlushAndEvict(wait: true);
    }

    // Takes a full (index + hybrid-log) checkpoint and waits for completion.
    private void Checkpoint()
    {
        _ = _store.TakeFullCheckpointAsync(CheckpointType.FoldOver)
                  .AsTask().GetAwaiter().GetResult();
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;

        Flush();
        Checkpoint();

        // Dispose every per-thread session created via ThreadLocal.
        foreach (var session in _sessions.Values)
            session.Dispose();
        _sessions.Dispose();

        _store.Dispose();
        _device.Dispose();
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    // Returns the smallest PageSizeBits such that the largest serialized layer
    // record fits within one FasterKV page (minimum 22 = 4 MB).
    private static int ComputePageSizeBits(NetworkConfig config)
    {
        int maxPayload = 0;
        for (int l = 0; l < config.LayerSizes.Length; l++)
        {
            int size  = config.LayerSizes[l];
            int lower = l > 0 ? config.LayerSizes[l - 1] : 0;
            int bytes = LayerSerializer.ComputeByteLength(size, lower, hasDiag: false, hasPrior: false);
            if (bytes > maxPayload) maxPayload = bytes;
        }

        // FasterKV wraps each value in a record header (~32 bytes total overhead).
        int recordSize = maxPayload + 32;
        int bits = 22;
        while ((1 << bits) < recordSize) bits++;
        return bits;
    }
}
