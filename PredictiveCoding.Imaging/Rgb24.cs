using System.Runtime.InteropServices;

namespace PredictiveCoding.Imaging;

/// <summary>
/// Packed 24-bit RGB pixel (one byte per channel, R first). Layout-compatible
/// with the raw buffer of <see cref="RgbImage"/> so rows can be reinterpreted
/// as <c>Span&lt;Rgb24&gt;</c> without copying.
/// </summary>
[StructLayout(LayoutKind.Sequential, Pack = 1)]
public struct Rgb24 : IEquatable<Rgb24>
{
    public byte R;
    public byte G;
    public byte B;

    public Rgb24(byte r, byte g, byte b)
    {
        R = r;
        G = g;
        B = b;
    }

    public readonly bool Equals(Rgb24 other) => R == other.R && G == other.G && B == other.B;

    public override readonly bool Equals(object? obj) => obj is Rgb24 other && Equals(other);

    public override readonly int GetHashCode() => (R << 16) | (G << 8) | B;

    public static bool operator ==(Rgb24 left, Rgb24 right) => left.Equals(right);

    public static bool operator !=(Rgb24 left, Rgb24 right) => !left.Equals(right);

    public override readonly string ToString() => $"Rgb24({R}, {G}, {B})";
}
