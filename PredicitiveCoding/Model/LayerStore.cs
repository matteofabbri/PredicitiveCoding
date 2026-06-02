using FASTER.core;
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

    // SpanByteFunctions<TKey, TOutput, TContext>:  Input=SpanByte, Output=SpanByteAndMemory
    private readonly ClientSession<
        int, SpanByte,
        SpanByte, SpanByteAndMemory,
        Empty, SpanByteFunctions<int, SpanByteAndMemory, Empty>> _session;

    // Write-through cache: FasterKV v2.6.5 does not reliably populate
    // SpanByteAndMemory for large records (>~256 bytes) via its synchronous
    // Read path, regardless of whether the record is in the mutable or
    // read-only region.  We keep a serialized copy of every layer in a
    // plain dictionary so that ReadLayer always gets correct data.
    // FasterKV is still written on every mutation for on-disk persistence.
    private readonly Dictionary<int, byte[]> _cache = [];

    // ── Construction ──────────────────────────────────────────────────────────

    /// <param name="storagePath">
    /// Directory for the log device.  When null the store uses an ephemeral
    /// temp file that is deleted on disposal.
    /// </param>
    public LayerStore(NetworkConfig config, Random random)
    {
        Configuration = config;
        Random = random;

        Directory.CreateDirectory(config.Path);

        _device = Devices.CreateLogDevice(
            Path.Combine(config.Path, "layers.log"),
            preallocateFile: false,
            deleteOnClose: false);

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
            Status status = _session.Read(ref index, ref output);

            if (status.IsPending)
            {
                _session.CompletePendingWithOutputs(out var iter, wait: true);
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

                _cache[index] = bytes.ToArray();
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
        PersistToStore(index, payload);
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
            _session.Upsert(ref index, ref sb);
        }
        finally
        {
            handle.Free();
        }
    }

    // ── Factory read ─────────────────────────────────────────────────────────

    /// <summary>
    /// Reads the stored bytes for <paramref name="index"/>, allocates a new
    /// <see cref="Layer"/> from the binary header, deserializes all fields into
    /// it, and returns it.  Returns <c>null</c> if the key is not present.
    /// </summary>
    public Layer ReadLayer(int index)
    {
        if (_cache.TryGetValue(index, out byte[]? cached))
            return LayerSerializer.AllocateAndDeserialize(cached, index, Configuration.Activation);

        // Cache miss (e.g. first access after a recovery that didn't populate
        // this slot) — initialize and persist.
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

    /// <summary>Serializes <paramref name="layer"/>, updates the cache, and upserts it by index.</summary>
    public void Write(int index, Layer layer)
    {
        byte[] payload = LayerSerializer.Serialize(layer);
        _cache[index] = payload;
        PersistToStore(index, payload);
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

    /// <summary>Forces all pending log writes to the device.</summary>
    public void Flush() => _store.Log.FlushAndEvict(wait: true);

    public void Dispose()
    {
        Flush();

        _session.Dispose();
        _store.Dispose();
        _device.Dispose();
    }
}
