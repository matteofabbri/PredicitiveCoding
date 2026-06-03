namespace PredictiveCoding.Expanded.MemoryNetwork;

// ─────────────────────────────────────────────────────────────────────────────
// MemoryBank  —  fixed-capacity associative key-value memory
//
// Storage:   circular buffer with FIFO eviction (oldest entry overwritten).
// Retrieval: cosine-similarity search, returns top-k entries sorted
//            descending by similarity.
// Keys and values are stored as float arrays (cloned on Write, returned
// read-only in MemoryEntry).
//
// Thread safety: all public operations are guarded by a single lock.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class MemoryBank
{
    private readonly float[]?[] _keys;
    private readonly float[]?[] _values;
    private int  _head;
    private int  _count;
    private readonly object _lock = new();

    public int Capacity { get; }
    public int KeyDim   { get; }
    public int ValueDim { get; }
    public int Count    { get { lock (_lock) return _count; } }

    public MemoryBank(int capacity, int keyDim, int valueDim)
    {
        if (capacity < 1) throw new ArgumentOutOfRangeException(nameof(capacity));
        Capacity = capacity;
        KeyDim   = keyDim;
        ValueDim = valueDim;
        _keys    = new float[]?[capacity];
        _values  = new float[]?[capacity];
    }

    // ── Write ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Stores a key/value pair.  When the bank is full the oldest entry is
    /// overwritten (FIFO).
    /// </summary>
    public void Write(float[] key, float[] value)
    {
        if (key.Length != KeyDim)
            throw new ArgumentException($"Key has {key.Length} dims; expected {KeyDim}.");
        if (value.Length != ValueDim)
            throw new ArgumentException($"Value has {value.Length} dims; expected {ValueDim}.");

        lock (_lock)
        {
            _keys[_head]   = (float[])key.Clone();
            _values[_head] = (float[])value.Clone();
            _head = (_head + 1) % Capacity;
            if (_count < Capacity) _count++;
        }
    }

    // ── Query ────────────────────────────────────────────────────────────────

    /// <summary>
    /// Returns the top-<paramref name="topK"/> entries sorted descending by
    /// cosine similarity to <paramref name="query"/>.
    /// Returns an empty array when the bank is empty.
    /// </summary>
    public MemoryEntry[] QueryTopK(float[] query, int topK)
    {
        if (query.Length != KeyDim)
            throw new ArgumentException($"Query has {query.Length} dims; expected {KeyDim}.");

        lock (_lock)
        {
            int n = _count;
            if (n == 0) return [];

            topK = Math.Min(topK, n);

            var scored = new (float sim, int idx)[n];
            for (int i = 0; i < n; i++)
                scored[i] = (CosineSimilarity(query, _keys[i]!  /* non-null: i < _count */), i);

            Array.Sort(scored, (a, b) => b.sim.CompareTo(a.sim));

            var result = new MemoryEntry[topK];
            for (int r = 0; r < topK; r++)
                result[r] = new MemoryEntry(_values[scored[r].idx]!, scored[r].sim);
            return result;
        }
    }

    // ── Helpers ──────────────────────────────────────────────────────────────

    private static float CosineSimilarity(float[] a, float[] b)
    {
        float dot = 0f, normA = 0f, normB = 0f;
        for (int i = 0; i < a.Length; i++)
        {
            dot   += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }
        float denom = MathF.Sqrt(normA) * MathF.Sqrt(normB);
        return denom < 1e-9f ? 0f : dot / denom;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MemoryEntry  —  a retrieved memory with its retrieval similarity score
// ─────────────────────────────────────────────────────────────────────────────

public record MemoryEntry(float[] Value, float Similarity);
