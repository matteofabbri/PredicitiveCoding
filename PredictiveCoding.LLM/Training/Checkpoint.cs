using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using FASTER.core;
using PredictiveCoding.LLM.Autograd;

namespace PredictiveCoding.LLM.Training;

// ─────────────────────────────────────────────────────────────────────────────
// Checkpoint  —  save / resume trained parameters via FASTER KV.
//
// Persists the model's parameter tensors through a FasterKV<int, SpanByte> store,
// the same persistence backend the core PredictiveCoding network uses (LayerStore).
// Weight persistence therefore runs on a single storage engine across the whole
// solution instead of a bespoke flat file.
//
// Key   : parameter index (int), in the fixed order returned by Model.Parameters().
// Value : flat blob [rows(4)][cols(4)][fp32 data] stored as a SpanByte whose
//         4-byte length header precedes the payload.
//
// Save() opens a fresh store, upserts every tensor, flushes the log and takes a
// full FASTER checkpoint. Load() opens the store, recovers that checkpoint, and
// copies each stored tensor back into the supplied parameters in place — tensor
// count and shapes must match.
//
// <paramref name="dir"/> is a directory owned exclusively by this checkpoint
// (it holds the FASTER log + checkpoint folder); Save() clears any prior store
// artifacts there so each write is pristine.
// ─────────────────────────────────────────────────────────────────────────────

public static class Checkpoint
{
    private const string LogName        = "weights.log";   // FASTER hybrid-log device
    private const string CheckpointsDir = "checkpoints";   // FASTER checkpoint folder

    public static void Save(string dir, IReadOnlyList<Tensor> parameters)
    {
        Directory.CreateDirectory(dir);
        ClearStore(dir);   // pristine write — drop stale log segments / checkpoints

        using var store = new TensorStore(dir, parameters, recover: false);
        for (int index = 0; index < parameters.Count; index++)
            store.UpsertTensor(index, parameters[index]);
        store.Commit();
    }

    /// <summary>Loads parameters in place; tensor count and shapes must match.</summary>
    public static void Load(string dir, IReadOnlyList<Tensor> parameters)
    {
        using var store = new TensorStore(dir, parameters, recover: true);

        for (int index = 0; index < parameters.Count; index++)
        {
            byte[] payload = store.Read(index)
                ?? throw new InvalidDataException(
                    $"Checkpoint under '{dir}' is missing tensor {index}; model has {parameters.Count} tensors.");
            Deserialize(payload, index, parameters[index]);
        }

        // A pristine store holds exactly the saved tensors at keys 0..Count-1;
        // anything at Count means the checkpoint belongs to a larger model.
        if (store.Read(parameters.Count) is not null)
            throw new InvalidDataException(
                $"Checkpoint under '{dir}' has more tensors than the model ({parameters.Count}).");
    }

    // ── Serialization ──────────────────────────────────────────────────────────

    // Reads [rows][cols][data] from a stored payload into target, validating shape.
    private static void Deserialize(byte[] payload, int index, Tensor target)
    {
        int rows = Unsafe.ReadUnaligned<int>(ref payload[0]);
        int cols = Unsafe.ReadUnaligned<int>(ref payload[sizeof(int)]);
        if (rows != target.Rows || cols != target.Cols)
            throw new InvalidDataException(
                $"Shape mismatch for tensor {index}: checkpoint [{rows},{cols}] vs model [{target.Rows},{target.Cols}].");
        Buffer.BlockCopy(payload, sizeof(int) * 2, target.Data, 0, target.Data.Length * sizeof(float));
    }

    // ── On-disk cleanup ──────────────────────────────────────────────────────────

    private static void ClearStore(string dir)
    {
        foreach (string file in Directory.EnumerateFiles(dir, LogName + "*"))
            File.Delete(file);
        string checkpoints = Path.Combine(dir, CheckpointsDir);
        if (Directory.Exists(checkpoints)) Directory.Delete(checkpoints, recursive: true);
    }

    // True when CheckpointDir already holds at least one completed checkpoint.
    private static bool HasCheckpoint(string checkpointDir)
    {
        string cpr = Path.Combine(checkpointDir, "cpr-checkpoints");
        return Directory.Exists(cpr) && Directory.EnumerateDirectories(cpr).Any();
    }

    // Smallest PageSizeBits such that the largest tensor record fits one FASTER
    // page (minimum 22 = 4 MB), mirroring LayerStore.ComputePageSizeBits.
    private static int ComputePageSizeBits(IReadOnlyList<Tensor> parameters)
    {
        int maxPayload = 0;
        foreach (var t in parameters)
        {
            int bytes = sizeof(int) * 2 + t.Data.Length * sizeof(float);
            if (bytes > maxPayload) maxPayload = bytes;
        }

        int recordSize = maxPayload + 32;   // FasterKV wraps each value in a record header
        int bits = 22;
        while ((1 << bits) < recordSize) bits++;
        return bits;
    }

    // ─────────────────────────────────────────────────────────────────────────
    // TensorStore  —  one-shot FasterKV<int, SpanByte> over a parameter list.
    //
    // Owns the log device, store and the single session for the save/load pass,
    // and disposes all three. The base SpanByteFunctions only ships write/RMW
    // callbacks, so Functions adds the read callbacks that populate the output
    // (exactly as LayerStore does).
    // ─────────────────────────────────────────────────────────────────────────
    private sealed class TensorStore : IDisposable
    {
        private sealed class Functions : SpanByteFunctions<int, SpanByteAndMemory, Empty>
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
        private readonly ClientSession<int, SpanByte, SpanByte, SpanByteAndMemory, Empty, Functions> _session;

        public TensorStore(string dir, IReadOnlyList<Tensor> parameters, bool recover)
        {
            string checkpointDir = Path.Combine(dir, CheckpointsDir);
            _device = Devices.CreateLogDevice(
                Path.Combine(dir, LogName),
                preallocateFile: false,
                deleteOnClose: false);

            int pageSizeBits   = ComputePageSizeBits(parameters);
            int memorySizeBits = pageSizeBits + 4;   // 16 pages in memory

            _store = new FasterKV<int, SpanByte>(
                size: 1L << 20,
                logSettings: new LogSettings
                {
                    LogDevice      = _device,
                    PageSizeBits   = pageSizeBits,
                    MemorySizeBits = memorySizeBits,
                },
                checkpointSettings: new CheckpointSettings { CheckpointDir = checkpointDir });

            if (recover)
            {
                if (!HasCheckpoint(checkpointDir))
                    throw new InvalidDataException($"No LLM checkpoint found under '{dir}'.");
                _store.Recover();
            }

            _session = _store.For(new Functions()).NewSession<Functions>();
        }

        // Stores [rows][cols][data] for one tensor under its index. The pinned
        // buffer is laid out as [spanLen][rows][cols][data]; SpanByte.Reinterpret
        // reads the leading int as the value length.
        public void UpsertTensor(int index, Tensor t)
        {
            int dataBytes  = t.Data.Length * sizeof(float);
            int payloadLen = sizeof(int) * 2 + dataBytes;
            byte[] buf = new byte[sizeof(int) + payloadLen];

            Unsafe.WriteUnaligned(ref buf[0],               payloadLen);
            Unsafe.WriteUnaligned(ref buf[sizeof(int)],     t.Rows);
            Unsafe.WriteUnaligned(ref buf[sizeof(int) * 2], t.Cols);
            Buffer.BlockCopy(t.Data, 0, buf, sizeof(int) * 3, dataBytes);

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

        // Returns the stored [rows][cols][data] payload for index, or null if absent.
        public byte[]? Read(int index)
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

        // Flushes the log to disk and takes a full checkpoint so a later process can recover.
        public void Commit()
        {
            _store.Log.FlushAndEvict(wait: true);
            _ = _store.TakeFullCheckpointAsync(CheckpointType.FoldOver)
                      .AsTask().GetAwaiter().GetResult();
        }

        public void Dispose()
        {
            _session.Dispose();
            _store.Dispose();
            _device.Dispose();
        }
    }
}
