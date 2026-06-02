using FASTER.core;
using PredictiveCoding.Networks;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

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
    private readonly NetworkConfig Configuration;
    private readonly Random Random;

    // FasterKV is thread-safe at the store level; sessions are NOT.
    // Each thread gets its own session created on first use via ThreadLocal.
    // The store itself is shared and lock-free internally.
    private readonly ThreadLocal<ClientSession<
        int, SpanByte,
        SpanByte, SpanByteAndMemory,
        Empty, SpanByteFunctions<int, SpanByteAndMemory, Empty>>> _sessions;

    private ClientSession<int, SpanByte, SpanByte, SpanByteAndMemory, Empty,
        SpanByteFunctions<int, SpanByteAndMemory, Empty>> Session => _sessions.Value!;

    // Write-through cache: FasterKV v2.6.5 does not reliably populate
    // SpanByteAndMemory for large records (>~256 bytes) via its synchronous
    // Read path, regardless of whether the record is in the mutable or
    // read-only region.  We keep a serialized copy of every layer in a
    // plain dictionary so that ReadLayer always gets correct data.
    // FasterKV is still written on every mutation for on-disk persistence
    // when _lazyPersistence is false.  When true, FasterKV writes are
    // deferred until Flush() — useful for large networks where per-sample
    // persistence would saturate I/O (e.g. CIFAR-10 with 6 MB weight matrices).
    //
    // Thread safety: _cache, _resident and _dirtyKeys are protected by
    // Network's ReaderWriterLockSlim (all callers hold it before entering
    // LayerStore).  No additional lock is needed here.
    private readonly Dictionary<int, byte[]> _cache = [];
    private readonly bool _lazyPersistence;
    private readonly HashSet<int> _dirtyKeys = [];
    private int _disposed = 0;

    // In lazy mode, Layer objects are kept in memory so that ReadLayer can
    // return a clone without byte[] serialization/deserialization overhead.
    // This eliminates ~6 MB of GC-allocated arrays per training sample when
    // working with large weight matrices (e.g. CIFAR-10, 3072×512 weights).
    private readonly Dictionary<int, Layer>? _resident;

    // ── Construction ──────────────────────────────────────────────────────────

    /// <param name="storagePath">
    /// Directory for the log device.  When null the store uses an ephemeral
    /// temp file that is deleted on disposal.
    /// </param>
    public LayerStore(NetworkConfig config, Random random, bool lazyPersistence = false)
    {
        Configuration    = config;
        Random           = random;
        _lazyPersistence = lazyPersistence;
        if (_lazyPersistence) _resident = [];

        Directory.CreateDirectory(config.Path);

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
            });

        _sessions = new ThreadLocal<ClientSession<
                int, SpanByte, SpanByte, SpanByteAndMemory, Empty,
                SpanByteFunctions<int, SpanByteAndMemory, Empty>>>(
            () => _store
                .For(new SpanByteFunctions<int, SpanByteAndMemory, Empty>())
                .NewSession<SpanByteFunctions<int, SpanByteAndMemory, Empty>>(),
            trackAllValues: true);

        if (Directory.GetFiles(config.Path, "*.*", SearchOption.AllDirectories).Any())
        {
            _store.Recover();
            PopulateCacheFromStore();
            return;
        }

        // Storage is empty — initialize and immediately cache every layer.
        for (int index = 0; index < Configuration.LayerSizes.Length; index++)
        {
            var layer = Initialize(index);
            CacheAndPersist(index, layer);
        }
    }

    // Reads all layer records from FasterKV into the in-memory cache.
    // Called once after Recover() so that subsequent ReadLayer calls are
    // served from the cache without touching FasterKV's read path.
    private void PopulateCacheFromStore()
    {
        for (int index = 0; index < Configuration.LayerSizes.Length; index++)
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
                continue; // layer absent; ReadLayer will initialize it on first access

            try
            {
                ReadOnlySpan<byte> bytes = output.IsSpanByte
                    ? output.SpanByte.AsReadOnlySpan()
                    : output.Memory!.Memory.Span[..output.Length];

                byte[] arr = bytes.ToArray();
                _cache[index] = arr;
                if (_lazyPersistence)
                    _resident![index] = LayerSerializer.AllocateAndDeserialize(
                        arr, index, Configuration.Activation);
            }
            finally
            {
                if (!output.IsSpanByte) output.Memory?.Dispose();
            }
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

    private void CacheAndPersist(int index, Layer layer)
    {
        byte[] payload = LayerSerializer.Serialize(layer);
        _cache[index] = payload;
        PersistToStore(index, payload);   // always write at init time
        if (_lazyPersistence)
            _resident![index] = layer;    // keep live object for fast reads
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

    // ── Factory read ─────────────────────────────────────────────────────────

    /// <summary>
    /// Returns a fresh <see cref="Layer"/> for <paramref name="index"/>.
    /// In lazy mode: clones the resident Layer object (no byte[] round-trip).
    /// In eager mode: deserializes from the byte[] cache (original behaviour).
    /// </summary>
    public Layer ReadLayer(int index)
    {
        if (_lazyPersistence && _resident!.TryGetValue(index, out Layer? resident))
            return resident.Clone();

        if (_cache.TryGetValue(index, out byte[]? cached))
            return LayerSerializer.AllocateAndDeserialize(cached, index, Configuration.Activation);

        // Cache miss — initialize and persist.
        var newLayer = Initialize(index);
        CacheAndPersist(index, newLayer);
        return newLayer;
    }

    /// <summary>
    /// Loads all <c>config.LayerSizes.Length</c> layers from the store and
    /// returns them as a freshly allocated array.
    /// Throws <see cref="InvalidOperationException"/> if any layer is missing.
    /// </summary>
    public Layer[] ReadAll()
    {
        var layers = new Layer[Configuration.LayerSizes.Length];
        for (int l = 0; l < layers.Length; l++)
            layers[l] = ReadLayer(l)
                ?? throw new InvalidOperationException(
                    $"Layer {l} not found in store — call WriteAll before ReadAll.");
        return layers;
    }

    // ── Write ───────────────────────────────────────────────────────────────

    /// <summary>
    /// Persists <paramref name="layer"/> to the store.
    /// Lazy mode: stores the live Layer object in the resident cache and defers
    ///            serialization + FasterKV write until <see cref="Flush"/>.
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
            byte[] payload = LayerSerializer.Serialize(layer);
            _cache[index] = payload;
            PersistToStore(index, payload);
        }
    }

    /// <summary>Checkpoints every layer in <paramref name="layers"/>.</summary>
    public void WriteAll(IReadOnlyList<Layer> layers)
    {
        for (int index = 0; index < layers.Count; index++)
        {
            var layer = layers[index];
            Write(index, layer);
        }
    }

    /// <summary>
    /// Flushes dirty resident layers to the byte[] cache and FasterKV
    /// (no-op in eager mode), then forces the device log.
    /// Call once per epoch when <see cref="NetworkConfig.LazyPersistence"/> is true.
    /// </summary>
    public void Flush()
    {
        if (_lazyPersistence)
        {
            foreach (int idx in _dirtyKeys)
            {
                byte[] payload = LayerSerializer.Serialize(_resident![idx]);
                _cache[idx] = payload;
                PersistToStore(idx, payload);
            }
            _dirtyKeys.Clear();
        }
        _store.Log.FlushAndEvict(wait: true);
    }

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0) return;

        Flush();

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
