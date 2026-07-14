using PredictiveCoding.Imaging;
using PredictiveCoding.Vision.Detectors;
using PredictiveCoding.Vision.Tracking;

namespace PredictiveCoding.Vision;

/// <summary>
/// Minimal, dependency-free box overlay for saved frames — draws rectangle
/// outlines directly on the pixels (no font/drawing package needed).
/// </summary>
public static class Annotate
{
    private static readonly Rgb24 DetColor = new(0, 255, 0);    // green
    private static readonly Rgb24 LineColor = new(255, 220, 32); // amber tripwire
    private static readonly Rgb24 ZoneColor = new(32, 230, 230); // cyan zone

    // Per-track palette: same colour ⇒ same tracked person across frames.
    private static readonly Rgb24[] Palette =
    [
        new(255,  64,  64), new( 64, 255,  64), new( 64, 160, 255), new(255, 220,  32),
        new(255,  64, 255), new( 32, 230, 230), new(255, 150,  40), new(170, 110, 255),
    ];

    /// <summary>Draws a green outline around every raw detection (in place).</summary>
    public static void DrawBoxes(RgbImage image, IReadOnlyList<Detection> detections, int thickness = 2)
    {
        foreach (Detection d in detections)
            DrawRect(image, (int)d.X1, (int)d.Y1, (int)d.X2, (int)d.Y2, thickness, DetColor);
    }

    /// <summary>Draws each tracked person's box in a stable per-id colour (in place).</summary>
    public static void DrawTracks(RgbImage image, IReadOnlyList<Track> tracks, int thickness = 2)
    {
        foreach (Track t in tracks)
        {
            Rgb24 c = Palette[(t.Id % Palette.Length + Palette.Length) % Palette.Length];
            DrawRect(image, (int)t.X1, (int)t.Y1, (int)t.X2, (int)t.Y2, thickness, c);
        }
    }

    /// <summary>Draws a virtual tripwire line A→B (in place).</summary>
    public static void DrawLine(RgbImage image, float ax, float ay, float bx, float by,
        Rgb24? color = null, int thickness = 2)
    {
        Rgb24 c = color ?? LineColor;
        float dx = bx - ax, dy = by - ay;
        float len = MathF.Sqrt(dx * dx + dy * dy);
        if (len < 1e-3f) return;

        float px = -dy / len, py = dx / len;   // unit perpendicular, for thickness
        int half = Math.Max(0, thickness / 2);
        for (int o = -half; o <= half; o++)
            DrawSegment(image,
                (int)MathF.Round(ax + px * o), (int)MathF.Round(ay + py * o),
                (int)MathF.Round(bx + px * o), (int)MathF.Round(by + py * o), c);
    }

    /// <summary>Draws the outline of a polygonal zone (auto-closed, in place).</summary>
    public static void DrawZone(RgbImage image, IReadOnlyList<(float X, float Y)> polygon,
        Rgb24? color = null, int thickness = 2)
    {
        ArgumentNullException.ThrowIfNull(polygon);
        if (polygon.Count < 2) return;
        Rgb24 c = color ?? ZoneColor;
        for (int i = 0; i < polygon.Count; i++)
        {
            var a = polygon[i];
            var b = polygon[(i + 1) % polygon.Count];
            DrawLine(image, a.X, a.Y, b.X, b.Y, c, thickness);
        }
    }

    // Bresenham line.
    private static void DrawSegment(RgbImage img, int x0, int y0, int x1, int y1, Rgb24 c)
    {
        int dx = Math.Abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
        int dy = -Math.Abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
        int err = dx + dy;
        while (true)
        {
            if ((uint)x0 < (uint)img.Width && (uint)y0 < (uint)img.Height) img[x0, y0] = c;
            if (x0 == x1 && y0 == y1) break;
            int e2 = 2 * err;
            if (e2 >= dy) { err += dy; x0 += sx; }
            if (e2 <= dx) { err += dx; y0 += sy; }
        }
    }

    private static void DrawRect(RgbImage img, int x1, int y1, int x2, int y2, int t, Rgb24 c)
    {
        x1 = Math.Clamp(x1, 0, img.Width  - 1);
        y1 = Math.Clamp(y1, 0, img.Height - 1);
        x2 = Math.Clamp(x2, 0, img.Width  - 1);
        y2 = Math.Clamp(y2, 0, img.Height - 1);
        for (int k = 0; k < t; k++)
        {
            HLine(img, x1, x2, y1 + k, c);
            HLine(img, x1, x2, y2 - k, c);
            VLine(img, y1, y2, x1 + k, c);
            VLine(img, y1, y2, x2 - k, c);
        }
    }

    private static void HLine(RgbImage img, int xa, int xb, int y, Rgb24 c)
    {
        if (y < 0 || y >= img.Height) return;
        int lo = Math.Max(0, xa), hi = Math.Min(img.Width - 1, xb);
        for (int x = lo; x <= hi; x++) img[x, y] = c;
    }

    private static void VLine(RgbImage img, int ya, int yb, int x, Rgb24 c)
    {
        if (x < 0 || x >= img.Width) return;
        int lo = Math.Max(0, ya), hi = Math.Min(img.Height - 1, yb);
        for (int y = lo; y <= hi; y++) img[x, y] = c;
    }
}
