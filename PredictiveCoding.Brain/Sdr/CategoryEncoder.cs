namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// CategoryEncoder  —  discrete category → SDR encoder.
//
// Each category gets its own disjoint block of W active bits, so distinct
// categories have zero overlap (maximally separable) while the same category
// always maps to the identical SDR. Use for unordered symbols (object ids,
// labels, discrete sensor channels).
// ─────────────────────────────────────────────────────────────────────────────

public sealed class CategoryEncoder : ISdrEncoder<int>
{
    private readonly int _categories;
    private readonly int _w;

    public int Size => _categories * _w;

    /// <param name="categories">Number of distinct categories.</param>
    /// <param name="w">Active bits per category (the block width).</param>
    public CategoryEncoder(int categories, int w)
    {
        if (categories <= 0) throw new ArgumentOutOfRangeException(nameof(categories), "categories must be positive.");
        if (w <= 0) throw new ArgumentOutOfRangeException(nameof(w), "w must be positive.");
        _categories = categories;
        _w = w;
    }

    public Sdr Encode(int category)
    {
        if (category < 0 || category >= _categories)
            throw new ArgumentOutOfRangeException(nameof(category), $"category must be in [0, {_categories}).");

        int start = category * _w;
        var bits = new int[_w];
        for (int i = 0; i < _w; i++) bits[i] = start + i;
        return new Sdr(Size, bits);
    }
}
