using System.Globalization;

namespace PredictiveCoding.Brain;

// ─────────────────────────────────────────────────────────────────────────────
// SdrCodec  —  compact, lossless text encoding for SDRs.
//
// An SDR is fully described by its size and its active-bit indices, so it
// serialises to "size|b0,b1,…". This is the leaf format the object-model store
// uses to persist learned features and locations; it round-trips exactly,
// including empty (zero-cardinality) SDRs.
// ─────────────────────────────────────────────────────────────────────────────

public static class SdrCodec
{
    /// <summary>Encodes an SDR as <c>"size|b0,b1,…"</c> (no active bits ⇒ <c>"size|"</c>).</summary>
    public static string Encode(Sdr sdr)
    {
        ArgumentNullException.ThrowIfNull(sdr);
        return $"{sdr.Size.ToString(CultureInfo.InvariantCulture)}|{string.Join(",", sdr.ActiveBits)}";
    }

    /// <summary>Parses a string produced by <see cref="Encode"/> back into an SDR.</summary>
    public static Sdr Decode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        int bar = text.IndexOf('|');
        if (bar < 0) throw new FormatException($"Malformed SDR (no '|'): '{text}'.");

        int size = int.Parse(text.AsSpan(0, bar), CultureInfo.InvariantCulture);
        string body = text[(bar + 1)..];
        if (body.Length == 0) return Sdr.Empty(size);

        string[] parts = body.Split(',');
        var bits = new int[parts.Length];
        for (int i = 0; i < parts.Length; i++)
            bits[i] = int.Parse(parts[i], CultureInfo.InvariantCulture);
        return new Sdr(size, bits);
    }
}
