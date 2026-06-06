namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// Sdr  —  Sparse Distributed Representation.
//
// A fixed-length binary vector with a small fraction of active bits, stored as a
// sorted, distinct list of active indices. SDRs are the lingua franca of the
// Thousand Brains theory: information is carried by *which* bits are on, overlap
// measures similarity, and high dimensionality + sparsity make them robust to
// noise and give enormous representational capacity.
//
// Reference: Hawkins & Ahmad (2016); Ahmad & Hawkins (2016) "Properties of SDRs".
// ─────────────────────────────────────────────────────────────────────────────

public sealed class Sdr : IEquatable<Sdr>
{
    private readonly int[] _active;   // sorted, distinct, each in [0, Size)

    /// <summary>Total number of bits (the dimensionality).</summary>
    public int Size { get; }

    /// <summary>The active bit indices, ascending.</summary>
    public IReadOnlyList<int> ActiveBits => _active;

    /// <summary>Number of active bits.</summary>
    public int Cardinality => _active.Length;

    /// <summary>Fraction of bits that are active.</summary>
    public float Sparsity => Size == 0 ? 0f : (float)_active.Length / Size;

    /// <summary>Creates an SDR of <paramref name="size"/> bits with the given active indices
    /// (deduplicated and sorted; every index must be in <c>[0, size)</c>).</summary>
    public Sdr(int size, IEnumerable<int> activeBits)
    {
        if (size <= 0) throw new ArgumentOutOfRangeException(nameof(size), "Size must be positive.");
        ArgumentNullException.ThrowIfNull(activeBits);
        Size = size;

        int[] arr = activeBits as int[] is { } a ? (int[])a.Clone() : activeBits.ToArray();
        Array.Sort(arr);

        int w = 0;
        for (int i = 0; i < arr.Length; i++)
            if (i == 0 || arr[i] != arr[i - 1]) arr[w++] = arr[i];
        if (w != arr.Length) Array.Resize(ref arr, w);

        if (arr.Length > 0 && (arr[0] < 0 || arr[^1] >= size))
            throw new ArgumentOutOfRangeException(nameof(activeBits),
                $"Active bit index out of range [0,{size}).");

        _active = arr;
    }

    /// <summary>True if bit <paramref name="bit"/> is active.</summary>
    public bool IsActive(int bit) => Array.BinarySearch(_active, bit) >= 0;

    // ── Similarity ───────────────────────────────────────────────────────────

    /// <summary>Number of bits active in both SDRs (the core similarity measure).</summary>
    public int Overlap(Sdr other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (other.Size != Size)
            throw new ArgumentException($"Size mismatch: {Size} vs {other.Size}.", nameof(other));

        int i = 0, j = 0, count = 0;
        int[] a = _active, b = other._active;
        while (i < a.Length && j < b.Length)
        {
            if (a[i] == b[j]) { count++; i++; j++; }
            else if (a[i] < b[j]) i++;
            else j++;
        }
        return count;
    }

    /// <summary>Overlap normalised by the smaller cardinality, in <c>[0,1]</c>.</summary>
    public float OverlapScore(Sdr other)
    {
        int min = Math.Min(Cardinality, other.Cardinality);
        return min == 0 ? 0f : (float)Overlap(other) / min;
    }

    /// <summary>True if the two SDRs share at least <paramref name="minOverlap"/> active bits.</summary>
    public bool Matches(Sdr other, int minOverlap) => Overlap(other) >= minOverlap;

    // ── Set algebra ──────────────────────────────────────────────────────────

    /// <summary>Bitwise OR.</summary>
    public Sdr Union(Sdr other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (other.Size != Size)
            throw new ArgumentException($"Size mismatch: {Size} vs {other.Size}.", nameof(other));

        var set = new HashSet<int>(_active);
        set.UnionWith(other._active);
        return new Sdr(Size, set);
    }

    /// <summary>Bitwise AND.</summary>
    public Sdr Intersect(Sdr other)
    {
        ArgumentNullException.ThrowIfNull(other);
        if (other.Size != Size)
            throw new ArgumentException($"Size mismatch: {Size} vs {other.Size}.", nameof(other));

        int i = 0, j = 0;
        int[] a = _active, b = other._active;
        var shared = new List<int>();
        while (i < a.Length && j < b.Length)
        {
            if (a[i] == b[j]) { shared.Add(a[i]); i++; j++; }
            else if (a[i] < b[j]) i++;
            else j++;
        }
        return new Sdr(Size, shared);
    }

    // ── Transformations ──────────────────────────────────────────────────────

    /// <summary>
    /// Returns a copy with a <paramref name="fraction"/> of its active bits moved to
    /// previously-inactive positions. Cardinality is preserved, and the new bits are
    /// disjoint from the original active set, so overlap with the original is exactly
    /// <c>Cardinality − movedBits</c>.
    /// </summary>
    public Sdr AddNoise(float fraction, Random rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        if (fraction <= 0f || Cardinality == 0) return this;

        int k = (int)Math.Round(fraction * Cardinality);
        k = Math.Min(k, Math.Min(Cardinality, Size - Cardinality));
        if (k <= 0) return this;

        var original = new HashSet<int>(_active);
        var removed = new HashSet<int>(SampleWithoutReplacement(_active, k, rng));

        var result = new HashSet<int>(Cardinality);
        foreach (int bit in _active)
            if (!removed.Contains(bit)) result.Add(bit);

        int added = 0;
        while (added < k)
        {
            int candidate = rng.Next(Size);
            if (!original.Contains(candidate) && result.Add(candidate)) added++;
        }
        return new Sdr(Size, result);
    }

    /// <summary>A random subset of <paramref name="n"/> of the active bits.</summary>
    public Sdr Subsample(int n, Random rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        if (n >= Cardinality) return this;
        return new Sdr(Size, SampleWithoutReplacement(_active, Math.Max(0, n), rng));
    }

    // ── Dense conversions ────────────────────────────────────────────────────

    /// <summary>A dense <c>{0,1}</c> float vector of length <see cref="Size"/>.</summary>
    public float[] ToDense()
    {
        var dense = new float[Size];
        foreach (int bit in _active) dense[bit] = 1f;
        return dense;
    }

    /// <summary>Builds an SDR from a dense vector, activating bits whose value ≥ <paramref name="threshold"/>.</summary>
    public static Sdr FromDense(ReadOnlySpan<float> values, float threshold = 0.5f)
    {
        var bits = new List<int>();
        for (int i = 0; i < values.Length; i++)
            if (values[i] >= threshold) bits.Add(i);
        return new Sdr(values.Length == 0 ? 1 : values.Length, bits);
    }

    // ── Factories ────────────────────────────────────────────────────────────

    /// <summary>An empty SDR (no active bits).</summary>
    public static Sdr Empty(int size) => new(size, []);

    /// <summary>An SDR with the given active bits.</summary>
    public static Sdr FromActiveBits(int size, params int[] bits) => new(size, bits);

    /// <summary>A random SDR with exactly <paramref name="cardinality"/> active bits.</summary>
    public static Sdr Random(int size, int cardinality, Random rng)
    {
        ArgumentNullException.ThrowIfNull(rng);
        if (cardinality < 0 || cardinality > size)
            throw new ArgumentOutOfRangeException(nameof(cardinality), "Cardinality must be in [0, size].");

        var set = new HashSet<int>(cardinality);
        while (set.Count < cardinality) set.Add(rng.Next(size));
        return new Sdr(size, set);
    }

    // ── Equality ─────────────────────────────────────────────────────────────

    public bool Equals(Sdr? other)
        => other is not null && Size == other.Size && _active.AsSpan().SequenceEqual(other._active);

    public override bool Equals(object? obj) => Equals(obj as Sdr);

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(Size);
        hash.Add(_active.Length);
        foreach (int bit in _active) hash.Add(bit);
        return hash.ToHashCode();
    }

    public override string ToString() => $"Sdr({Cardinality}/{Size}: [{string.Join(",", _active)}])";

    // ── Helpers ──────────────────────────────────────────────────────────────

    // Fisher–Yates partial shuffle: returns k distinct elements drawn from source.
    private static int[] SampleWithoutReplacement(IReadOnlyList<int> source, int k, Random rng)
    {
        var copy = source.ToArray();
        k = Math.Min(k, copy.Length);
        for (int i = 0; i < k; i++)
        {
            int j = rng.Next(i, copy.Length);
            (copy[i], copy[j]) = (copy[j], copy[i]);
        }
        var result = new int[k];
        Array.Copy(copy, result, k);
        return result;
    }
}
