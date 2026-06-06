namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// ScalarEncoder  —  classic HTM scalar → SDR encoder.
//
// Maps a bounded scalar to a contiguous block of W active bits within N total
// bits. Nearby values share most of their active bits (high overlap); distant
// values share none — exactly the semantic-similarity property downstream
// sequence memory and columns rely on. Out-of-range values are clamped.
// ─────────────────────────────────────────────────────────────────────────────

public sealed class ScalarEncoder : ISdrEncoder<double>
{
    private readonly double _min;
    private readonly double _max;
    private readonly int _n;
    private readonly int _w;
    private readonly int _buckets;

    public int Size => _n;

    /// <param name="min">Lower bound of the input range.</param>
    /// <param name="max">Upper bound of the input range (≥ <paramref name="min"/>).</param>
    /// <param name="n">Total number of bits.</param>
    /// <param name="w">Number of active bits per value (the window width); must be &lt; <paramref name="n"/>.</param>
    public ScalarEncoder(double min, double max, int n, int w)
    {
        if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n), "n must be positive.");
        if (w <= 0 || w >= n) throw new ArgumentOutOfRangeException(nameof(w), "w must be in (0, n).");
        if (max < min) throw new ArgumentException("max must be >= min.", nameof(max));

        _min = min;
        _max = max;
        _n = n;
        _w = w;
        _buckets = n - w + 1;   // number of distinct window positions
    }

    public Sdr Encode(double value)
    {
        double clamped = Math.Clamp(value, _min, _max);
        double frac = _max > _min ? (clamped - _min) / (_max - _min) : 0.0;
        int start = (int)Math.Floor(frac * (_buckets - 1) + 0.5);
        start = Math.Clamp(start, 0, _buckets - 1);

        var bits = new int[_w];
        for (int i = 0; i < _w; i++) bits[i] = start + i;
        return new Sdr(_n, bits);
    }
}
